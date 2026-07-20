#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use typemach_agent::{ByteLimit, ExecLimits, ExecSpec, OpenFileLimit, PermissionProfile};

fn runtime_roots(program: &Path) -> Vec<PathBuf> {
    let mut roots = vec![program.to_path_buf()];
    for path in ["/nix/store", "/lib", "/lib64", "/usr/lib", "/usr/lib64"] {
        let path = PathBuf::from(path);
        if path.exists() {
            roots.push(path);
        }
    }
    roots
}

#[tokio::test]
async fn sandbox_dies_when_its_owning_process_is_killed() {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let temp = tempfile::tempdir().expect("orphan sandbox fixture");
    let helper = PathBuf::from(env!("CARGO_BIN_EXE_typemach-sandbox"));
    let mut owner = tokio::process::Command::new(&helper)
        .arg("__typemach_orphan_owner")
        .arg(temp.path())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn sandbox owner");
    let stdout = owner.stdout.take().expect("sandbox owner stdout");
    let mut lines = BufReader::new(stdout).lines();
    let sandbox_pid = tokio::time::timeout(Duration::from_secs(5), lines.next_line())
        .await
        .expect("sandbox owner reported child")
        .expect("read sandbox child pid")
        .expect("sandbox child pid line")
        .parse::<u32>()
        .expect("numeric sandbox child pid");
    owner.start_kill().expect("kill sandbox owner");
    owner.wait().await.expect("reap sandbox owner");

    let process_path = PathBuf::from(format!("/proc/{sandbox_pid}"));
    let gone = tokio::time::timeout(Duration::from_secs(5), async {
        while process_path.exists() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    if gone.is_err() {
        unsafe {
            libc::kill(sandbox_pid as libc::pid_t, libc::SIGKILL);
        }
    }
    assert!(gone.is_ok(), "sandbox survived its owning process");
}

#[tokio::test]
async fn sandbox_enforces_filesystem_network_and_write_boundaries() {
    let mut nofile_before = std::mem::MaybeUninit::<libc::rlimit>::uninit();
    assert_eq!(
        unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, nofile_before.as_mut_ptr()) },
        0
    );
    let nofile_before = unsafe { nofile_before.assume_init() };
    let temp = tempfile::tempdir().expect("sandbox fixture");
    let input_dir = temp.path().join("input");
    let scratch_dir = temp.path().join("scratch");
    let denied_dir = temp.path().join("denied");
    std::fs::create_dir_all(&input_dir).expect("input directory");
    std::fs::create_dir_all(&scratch_dir).expect("scratch directory");
    std::fs::create_dir_all(&denied_dir).expect("denied directory");
    let input = input_dir.join("input.txt");
    let denied = denied_dir.join("secret.txt");
    let output = scratch_dir.join("output.txt");
    std::fs::write(&input, "allowed").expect("input");
    std::fs::write(&denied, "secret").expect("denied fixture");

    let helper = PathBuf::from(env!("CARGO_BIN_EXE_typemach-sandbox"));
    let program = helper.canonicalize().expect("sandbox probe path");
    let spec = ExecSpec {
        program: program.clone(),
        args: vec![
            "__typemach_probe".into(),
            input.clone().into_os_string(),
            denied.clone().into_os_string(),
            output.clone().into_os_string(),
        ],
        cwd: scratch_dir.clone(),
        profile: PermissionProfile {
            read_only: runtime_roots(&program)
                .into_iter()
                .chain([input_dir.clone()])
                .collect(),
            writable: vec![scratch_dir.clone()],
            environment: BTreeMap::new(),
            limits: ExecLimits {
                cpu_time: Duration::from_secs(5),
                address_space: ByteLimit::new(512 * 1024 * 1024),
                file_size: ByteLimit::new(16 * 1024 * 1024),
                open_files: OpenFileLimit::new(64),
                max_processes: OpenFileLimit::new(32),
            },
        },
    };
    let mut child = spec.spawn(&helper).await.expect("spawn sandbox");
    let mut stdin = child.take_stdin().expect("sandbox stdin");
    stdin
        .write_all(b"business-frame")
        .await
        .expect("write stdin");
    drop(stdin);
    let mut stdout = child.take_stdout().expect("sandbox stdout");
    let mut stderr = child.take_stderr().expect("sandbox stderr");
    let stdout_task = tokio::spawn(async move {
        let mut output = String::new();
        stdout.read_to_string(&mut output).await.expect("stdout");
        output
    });
    let stderr_task = tokio::spawn(async move {
        let mut output = String::new();
        stderr.read_to_string(&mut output).await.expect("stderr");
        output
    });
    let status = child.wait().await.expect("wait for sandbox");
    let stdout = stdout_task.await.expect("stdout task");
    let stderr = stderr_task.await.expect("stderr task");

    assert!(status.success(), "sandbox failed: {stderr}");
    assert!(stdout.contains("allowed_read=true"));
    assert!(stdout.contains("denied_read=true"));
    assert!(stdout.contains("denied_input_write=true"));
    assert!(stdout.contains("denied_network=true"));
    assert!(stdout.contains("denied_namespace=true"));
    assert!(stdout.contains("denied_parent_prlimit=true"));
    assert!(stdout.contains("denied_queued_signal=true"));
    assert!(stdout.contains("denied_affinity=true"));
    assert!(stdout.contains("denied_memfd=true"));
    assert!(stdout.contains("output_write=true"));
    assert!(stdout.contains("business_stdin=true"));
    assert_eq!(
        std::fs::read_to_string(&input).expect("input unchanged"),
        "allowed"
    );
    assert_eq!(std::fs::read_to_string(&output).expect("output"), "written");
    let mut nofile_after = std::mem::MaybeUninit::<libc::rlimit>::uninit();
    assert_eq!(
        unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, nofile_after.as_mut_ptr()) },
        0
    );
    let nofile_after = unsafe { nofile_after.assume_init() };
    assert_eq!(nofile_after.rlim_cur, nofile_before.rlim_cur);
    assert_eq!(nofile_after.rlim_max, nofile_before.rlim_max);
}

#[tokio::test]
async fn sandbox_enforces_cpu_time_limit() {
    let temp = tempfile::tempdir().expect("CPU limit fixture");
    let helper = PathBuf::from(env!("CARGO_BIN_EXE_typemach-sandbox"));
    let program = helper.canonicalize().expect("sandbox probe path");
    let mut child = ExecSpec {
        program: program.clone(),
        args: vec!["__typemach_cpu_probe".into()],
        cwd: temp.path().to_path_buf(),
        profile: PermissionProfile {
            read_only: runtime_roots(&program),
            writable: vec![temp.path().to_path_buf()],
            environment: BTreeMap::new(),
            limits: ExecLimits {
                cpu_time: Duration::from_secs(1),
                address_space: ByteLimit::new(512 * 1024 * 1024),
                file_size: ByteLimit::new(16 * 1024 * 1024),
                open_files: OpenFileLimit::new(64),
                max_processes: OpenFileLimit::new(1),
            },
        },
    }
    .spawn(&helper)
    .await
    .expect("spawn CPU-bound sandbox");
    let status = tokio::time::timeout(Duration::from_secs(5), child.wait())
        .await
        .expect("CPU limit stopped sandbox")
        .expect("wait for CPU-bound sandbox");
    assert_eq!(status.signal(), Some(libc::SIGXCPU));
}
