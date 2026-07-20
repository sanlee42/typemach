use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};

const HELPER_ARG: &str = "__typemach_sandbox";
const MAX_REQUEST_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteLimit(u64);

impl ByteLimit {
    pub const fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenFileLimit(u64);

impl OpenFileLimit {
    pub const fn new(count: u64) -> Self {
        Self(count)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecLimits {
    pub cpu_time: Duration,
    pub address_space: ByteLimit,
    pub file_size: ByteLimit,
    pub open_files: OpenFileLimit,
    pub max_processes: OpenFileLimit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionProfile {
    pub read_only: Vec<PathBuf>,
    pub writable: Vec<PathBuf>,
    pub environment: BTreeMap<String, String>,
    pub limits: ExecLimits,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecSpec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub cwd: PathBuf,
    pub profile: PermissionProfile,
}

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("sandbox execution is supported only on Linux")]
    UnsupportedPlatform,
    #[error("sandbox helper invocation is missing its request")]
    MissingRequest,
    #[error("sandbox path must be absolute: {0}")]
    RelativePath(PathBuf),
    #[error("sandbox path does not exist: {0}")]
    MissingPath(PathBuf),
    #[error("encode or decode sandbox request failed")]
    Request(#[from] serde_json::Error),
    #[error("sandbox request exceeds limit")]
    RequestTooLarge,
    #[error("sandbox process operation failed")]
    Io(#[from] std::io::Error),
    #[cfg(target_os = "linux")]
    #[error("configure Landlock rules failed")]
    LandlockRuleset(#[from] landlock::RulesetError),
    #[cfg(target_os = "linux")]
    #[error("open a Landlock path failed")]
    LandlockPath(#[from] landlock::PathFdError),
    #[cfg(target_os = "linux")]
    #[error("configure seccomp rules failed")]
    Seccomp(#[from] seccompiler::Error),
    #[cfg(target_os = "linux")]
    #[error("install seccomp rules failed")]
    SeccompBackend(#[from] seccompiler::BackendError),
    #[error("Landlock did not enforce the requested filesystem policy")]
    LandlockNotEnforced,
    #[error("sandbox limit {name} is invalid: {value}")]
    InvalidLimit { name: &'static str, value: u64 },
}

pub struct ExecChild {
    child: Child,
    process_group: i32,
}

impl ExecSpec {
    pub async fn spawn(self, helper: &Path) -> Result<ExecChild, SandboxError> {
        validate_path(helper)?;
        validate_path(&self.program)?;
        validate_path(&self.cwd)?;
        for path in self
            .profile
            .read_only
            .iter()
            .chain(self.profile.writable.iter())
        {
            validate_path(path)?;
        }

        let request = serde_json::to_vec(&self)?;
        if request.len() > MAX_REQUEST_BYTES {
            return Err(SandboxError::RequestTooLarge);
        }
        let control_dir = tempfile::Builder::new()
            .prefix("typemach-sandbox-")
            .tempdir()?;
        let control_path = control_dir.path().join("control.sock");
        let listener = std::os::unix::net::UnixListener::bind(&control_path)?;
        listener.set_nonblocking(true)?;
        let mut command = Command::new(helper);
        #[cfg(unix)]
        {
            let owner_pid = std::process::id() as libc::pid_t;
            unsafe {
                command.pre_exec(move || {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) < 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::getppid() != owner_pid {
                        return Err(std::io::Error::other("sandbox owner exited during spawn"));
                    }
                    if libc::setsid() < 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        command
            .arg(HELPER_ARG)
            .arg(&control_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn()?;
        let started = std::time::Instant::now();
        let mut control = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if let Some(status) = child.try_wait()? {
                        return Err(SandboxError::Io(std::io::Error::other(format!(
                            "sandbox helper exited before handshake: {status}"
                        ))));
                    }
                    if started.elapsed() >= Duration::from_secs(5) {
                        let _ = child.start_kill();
                        let _ = child.wait().await;
                        return Err(SandboxError::Io(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            "sandbox helper handshake timed out",
                        )));
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                Err(error) => return Err(SandboxError::Io(error)),
            }
        };
        control.write_all(&(request.len() as u32).to_be_bytes())?;
        control.write_all(&request)?;
        control.flush()?;
        drop(control);
        drop(listener);
        drop(control_dir);
        let process_group = child.id().ok_or_else(|| {
            SandboxError::Io(std::io::Error::other("sandbox child has no process id"))
        })? as i32;
        Ok(ExecChild {
            child,
            process_group,
        })
    }
}

impl ExecChild {
    pub fn id(&self) -> Option<u32> {
        self.child.id()
    }

    pub fn take_stdin(&mut self) -> Option<ChildStdin> {
        self.child.stdin.take()
    }

    pub fn take_stdout(&mut self) -> Option<ChildStdout> {
        self.child.stdout.take()
    }

    pub fn take_stderr(&mut self) -> Option<ChildStderr> {
        self.child.stderr.take()
    }

    pub async fn wait(&mut self) -> Result<std::process::ExitStatus, SandboxError> {
        let status = self.child.wait().await?;
        // The leader may have exited while descendants remain in its group.
        // Reap the leader first, then terminate any residual descendants.
        kill_process_group(self.process_group);
        self.process_group = 0;
        Ok(status)
    }

    pub async fn kill(&mut self) -> Result<(), SandboxError> {
        kill_process_group(self.process_group);
        let _ = self.child.start_kill();
        let _ = self.child.wait().await?;
        self.process_group = 0;
        Ok(())
    }
}

impl Drop for ExecChild {
    fn drop(&mut self) {
        kill_process_group(self.process_group);
        let _ = self.child.start_kill();
    }
}

pub fn helper_requested() -> bool {
    std::env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new(HELPER_ARG))
}

pub fn run_sandbox_helper() -> Result<(), SandboxError> {
    if !helper_requested() {
        return Err(SandboxError::MissingRequest);
    }
    let control_path = std::env::args_os()
        .nth(2)
        .map(PathBuf::from)
        .ok_or(SandboxError::MissingRequest)?;
    let mut control = std::os::unix::net::UnixStream::connect(control_path)?;
    let mut frame = [0u8; 4];
    control.read_exact(&mut frame)?;
    let size = u32::from_be_bytes(frame) as usize;
    if size > MAX_REQUEST_BYTES {
        return Err(SandboxError::RequestTooLarge);
    }
    let mut request = vec![0u8; size];
    control.read_exact(&mut request)?;
    let spec: ExecSpec = serde_json::from_slice(&request)?;
    run_helper(spec)
}

fn validate_path(path: &Path) -> Result<(), SandboxError> {
    if !path.is_absolute() {
        return Err(SandboxError::RelativePath(path.to_path_buf()));
    }
    let metadata = path
        .metadata()
        .map_err(|_| SandboxError::MissingPath(path.to_path_buf()))?;
    if !metadata.is_dir() && !metadata.is_file() {
        return Err(SandboxError::MissingPath(path.to_path_buf()));
    }
    // Landlock resolves path rules at rule installation; reject symlinks so a
    // profile cannot silently grant a different target than the caller saw.
    if path.canonicalize().ok().as_deref() != Some(path) {
        return Err(SandboxError::MissingPath(path.to_path_buf()));
    }
    Ok(())
}

fn kill_process_group(process_group: i32) {
    if process_group <= 0 {
        return;
    }
    #[cfg(target_family = "unix")]
    unsafe {
        libc::kill(-process_group, libc::SIGKILL);
    }
}

#[cfg(target_os = "linux")]
fn run_helper(spec: ExecSpec) -> Result<(), SandboxError> {
    use std::os::unix::process::CommandExt;

    std::env::set_current_dir(&spec.cwd)?;
    // `ExecSpec::spawn` creates the session before this process is observable.
    // Keep the fallback for direct helper invocation.
    if unsafe { libc::getpgrp() } != unsafe { libc::getpid() } && unsafe { libc::setsid() } < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    apply_limits(&spec.profile.limits)?;
    set_no_new_privileges()?;
    apply_landlock(&spec.profile)?;
    apply_seccomp()?;

    let mut command = std::process::Command::new(&spec.program);
    command
        .args(&spec.args)
        .current_dir(&spec.cwd)
        .env_clear()
        .envs(&spec.profile.environment);
    Err(command.exec().into())
}

#[cfg(not(target_os = "linux"))]
fn run_helper(_spec: ExecSpec) -> Result<(), SandboxError> {
    Err(SandboxError::UnsupportedPlatform)
}

#[cfg(target_os = "linux")]
fn apply_limits(limits: &ExecLimits) -> Result<(), SandboxError> {
    set_cpu_limit(limits.cpu_time.as_secs())?;
    set_limit(
        libc::RLIMIT_AS,
        limits.address_space.get(),
        "address_space_bytes",
    )?;
    set_limit(
        libc::RLIMIT_FSIZE,
        limits.file_size.get(),
        "file_size_bytes",
    )?;
    set_limit(libc::RLIMIT_NOFILE, limits.open_files.get(), "open_files")?;
    set_limit(
        libc::RLIMIT_NPROC,
        limits.max_processes.get(),
        "max_processes",
    )?;
    set_limit(libc::RLIMIT_CORE, 0, "core_bytes")
}

#[cfg(target_os = "linux")]
fn set_cpu_limit(value: u64) -> Result<(), SandboxError> {
    if value == 0 {
        return Err(SandboxError::InvalidLimit {
            name: "cpu_time_seconds",
            value,
        });
    }
    let hard = value.checked_add(1).ok_or(SandboxError::InvalidLimit {
        name: "cpu_time_seconds",
        value,
    })?;
    set_limit_values(libc::RLIMIT_CPU, value, hard, "cpu_time_seconds")
}

#[cfg(all(target_os = "linux", any(target_env = "gnu", target_env = "uclibc")))]
type RlimitResource = libc::__rlimit_resource_t;

#[cfg(all(
    target_os = "linux",
    not(any(target_env = "gnu", target_env = "uclibc"))
))]
type RlimitResource = libc::c_int;

#[cfg(target_os = "linux")]
fn set_limit(resource: RlimitResource, value: u64, name: &'static str) -> Result<(), SandboxError> {
    if value == 0 && resource != libc::RLIMIT_CORE {
        return Err(SandboxError::InvalidLimit { name, value });
    }
    set_limit_values(resource, value, value, name)
}

#[cfg(target_os = "linux")]
fn set_limit_values(
    resource: RlimitResource,
    soft: u64,
    hard: u64,
    _name: &'static str,
) -> Result<(), SandboxError> {
    let limit = libc::rlimit {
        rlim_cur: soft as libc::rlim_t,
        rlim_max: hard as libc::rlim_t,
    };
    if unsafe { libc::setrlimit(resource, &limit) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn set_no_new_privileges() -> Result<(), SandboxError> {
    if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn apply_landlock(profile: &PermissionProfile) -> Result<(), SandboxError> {
    use landlock::{
        ABI, Access, AccessFs, CompatLevel, Compatible, Ruleset, RulesetAttr, RulesetCreatedAttr,
    };

    let abi = ABI::V4;
    let access_all = AccessFs::from_all(abi);
    let access_read = AccessFs::from_read(abi);
    let ruleset = Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(access_all)?
        .create()?
        .add_rules(landlock::path_beneath_rules(
            profile.read_only.iter(),
            access_read,
        ))?
        .add_rules(landlock::path_beneath_rules(
            profile.writable.iter(),
            access_all,
        ))?
        .no_new_privs(true);
    let status = ruleset.restrict_self()?;
    if status.ruleset == landlock::RulesetStatus::NotEnforced {
        return Err(SandboxError::LandlockNotEnforced);
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn apply_seccomp() -> Result<(), SandboxError> {
    use seccompiler::{
        BpfProgram, SeccompAction, SeccompFilter, SeccompRule, TargetArch, apply_filter,
    };

    let mut rules = BTreeMap::<i64, Vec<SeccompRule>>::new();
    for syscall in [
        libc::SYS_accept,
        libc::SYS_accept4,
        libc::SYS_bind,
        libc::SYS_connect,
        libc::SYS_getpeername,
        libc::SYS_getsockname,
        libc::SYS_getsockopt,
        libc::SYS_io_uring_enter,
        libc::SYS_io_uring_register,
        libc::SYS_io_uring_setup,
        libc::SYS_kill,
        libc::SYS_listen,
        libc::SYS_pidfd_open,
        libc::SYS_pidfd_getfd,
        libc::SYS_pidfd_send_signal,
        libc::SYS_prlimit64,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_prctl,
        libc::SYS_ptrace,
        libc::SYS_rt_sigqueueinfo,
        libc::SYS_rt_tgsigqueueinfo,
        libc::SYS_sched_setaffinity,
        libc::SYS_sched_setparam,
        libc::SYS_sched_setscheduler,
        libc::SYS_setpriority,
        libc::SYS_ioprio_set,
        libc::SYS_kcmp,
        libc::SYS_recvmmsg,
        libc::SYS_sendmmsg,
        libc::SYS_sendto,
        libc::SYS_setsockopt,
        libc::SYS_shutdown,
        libc::SYS_socket,
        libc::SYS_socketpair,
        libc::SYS_tkill,
        libc::SYS_tgkill,
        libc::SYS_clone,
        libc::SYS_clone3,
        libc::SYS_unshare,
        libc::SYS_setns,
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_pivot_root,
        libc::SYS_chroot,
        libc::SYS_bpf,
        libc::SYS_perf_event_open,
        libc::SYS_keyctl,
        libc::SYS_add_key,
        libc::SYS_request_key,
        libc::SYS_init_module,
        libc::SYS_finit_module,
        libc::SYS_delete_module,
        libc::SYS_userfaultfd,
        libc::SYS_memfd_create,
        libc::SYS_open_by_handle_at,
    ] {
        rules.insert(syscall, Vec::new());
    }

    let arch = if cfg!(target_arch = "x86_64") {
        TargetArch::x86_64
    } else if cfg!(target_arch = "aarch64") {
        TargetArch::aarch64
    } else {
        return Err(SandboxError::UnsupportedPlatform);
    };
    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow,
        SeccompAction::Errno(libc::EPERM as u32),
        arch,
    )?;
    let program: BpfProgram = filter.try_into()?;
    apply_filter(&program)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_marker_is_stable() {
        assert_eq!(HELPER_ARG, "__typemach_sandbox");
    }

    #[test]
    fn relative_paths_are_rejected_before_spawn() {
        let error = validate_path(Path::new("relative")).expect_err("relative path");
        assert!(matches!(error, SandboxError::RelativePath(_)));
    }
}
