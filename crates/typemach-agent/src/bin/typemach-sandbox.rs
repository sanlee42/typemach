fn main() {
    if std::env::args().nth(1).as_deref() == Some("__typemach_orphan_owner") {
        run_orphan_owner();
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("__typemach_sleep_probe") {
        std::thread::sleep(std::time::Duration::from_secs(60));
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("__typemach_cpu_probe") {
        loop {
            std::hint::spin_loop();
        }
    }
    if std::env::args().nth(1).as_deref() == Some("__typemach_probe") {
        run_probe();
        return;
    }
    if !typemach_agent::helper_requested() {
        eprintln!("typemach-sandbox is an internal execution helper");
        std::process::exit(2);
    }
    if let Err(error) = typemach_agent::run_sandbox_helper() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run_orphan_owner() {
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Duration;
    use typemach_agent::{ByteLimit, ExecLimits, ExecSpec, OpenFileLimit, PermissionProfile};

    let scratch = PathBuf::from(std::env::args().nth(2).expect("orphan probe scratch"));
    let helper = std::env::current_exe()
        .expect("orphan probe executable")
        .canonicalize()
        .expect("orphan probe canonical executable");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("orphan probe runtime");
    let child = runtime
        .block_on(
            ExecSpec {
                program: helper.clone(),
                args: vec!["__typemach_sleep_probe".into()],
                cwd: scratch.clone(),
                profile: PermissionProfile {
                    read_only: [
                        helper.clone(),
                        PathBuf::from("/nix/store"),
                        PathBuf::from("/lib"),
                        PathBuf::from("/lib64"),
                        PathBuf::from("/usr/lib"),
                        PathBuf::from("/usr/lib64"),
                    ]
                    .into_iter()
                    .filter(|path| path.exists())
                    .collect(),
                    writable: vec![scratch],
                    environment: BTreeMap::new(),
                    limits: ExecLimits {
                        cpu_time: Duration::from_secs(5),
                        address_space: ByteLimit::new(512 * 1024 * 1024),
                        file_size: ByteLimit::new(16 * 1024 * 1024),
                        open_files: OpenFileLimit::new(64),
                        max_processes: OpenFileLimit::new(1),
                    },
                },
            }
            .spawn(&helper),
        )
        .expect("spawn orphan probe");
    println!("{}", child.id().expect("orphan probe pid"));
    std::io::stdout().flush().expect("flush orphan probe pid");
    std::mem::forget(child);
    std::thread::sleep(Duration::from_secs(60));
}

fn run_probe() {
    use std::io::Read;

    let mut args = std::env::args().skip(2);
    let input = args.next().expect("probe input");
    let denied = args.next().expect("probe denied path");
    let output = args.next().expect("probe output");

    let allowed_read = std::fs::read_to_string(&input).is_ok_and(|value| value == "allowed");
    let denied_read = std::fs::read_to_string(&denied).is_err();
    let denied_input_write = std::fs::write(&input, "changed").is_err();
    let denied_network = std::net::TcpStream::connect("127.0.0.1:9").is_err();
    let denied_namespace = unsafe { libc::unshare(libc::CLONE_NEWNS) } == -1;
    let parent = unsafe { libc::getppid() };
    let lowered = libc::rlimit {
        rlim_cur: 1,
        rlim_max: 1,
    };
    let denied_parent_prlimit = denied_syscall(unsafe {
        libc::syscall(
            libc::SYS_prlimit64,
            parent,
            libc::RLIMIT_NOFILE,
            &lowered,
            std::ptr::null_mut::<libc::rlimit>(),
        )
    });
    let denied_queued_signal = denied_syscall(unsafe {
        libc::syscall(
            libc::SYS_rt_sigqueueinfo,
            parent,
            0,
            std::ptr::null::<libc::siginfo_t>(),
        )
    });
    let denied_affinity = denied_syscall(unsafe {
        libc::syscall(
            libc::SYS_sched_setaffinity,
            parent,
            0,
            std::ptr::null::<libc::cpu_set_t>(),
        )
    });
    let denied_memfd = denied_syscall(unsafe {
        libc::syscall(libc::SYS_memfd_create, c"typemach-probe".as_ptr(), 0)
    });
    let output_write = std::fs::write(&output, "written").is_ok();
    let mut stdin = String::new();
    let business_stdin =
        std::io::stdin().read_to_string(&mut stdin).is_ok() && stdin == "business-frame";

    println!(
        "allowed_read={allowed_read} denied_read={denied_read} denied_input_write={denied_input_write} denied_network={denied_network} denied_namespace={denied_namespace} denied_parent_prlimit={denied_parent_prlimit} denied_queued_signal={denied_queued_signal} denied_affinity={denied_affinity} denied_memfd={denied_memfd} output_write={output_write} business_stdin={business_stdin}"
    );
    if !(allowed_read
        && denied_read
        && denied_input_write
        && denied_network
        && denied_namespace
        && denied_parent_prlimit
        && denied_queued_signal
        && denied_affinity
        && denied_memfd
        && output_write
        && business_stdin)
    {
        std::process::exit(1);
    }
}

fn denied_syscall(result: libc::c_long) -> bool {
    result == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}
