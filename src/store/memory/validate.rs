use super::*;

pub(super) fn scope_key<Scope>(scope: &Scope) -> Result<String, MachineError>
where
    Scope: Serialize,
{
    let value = serde_json::to_value(scope).map_err(MachineError::Serialization)?;
    serde_json::to_string(&value).map_err(MachineError::Serialization)
}

pub(super) fn validate_event_run<E: RunEvent>(
    run_id: &RunId,
    session_id: &SessionId,
    event: &E,
) -> Result<(), MachineError> {
    if event.run_id() != run_id {
        return Err(MachineError::InvalidRunEvent {
            reason: "event run_id does not match target run".to_string(),
        });
    }
    if event.session_id() != session_id {
        return Err(MachineError::InvalidRunEvent {
            reason: "event session_id does not match target run".to_string(),
        });
    }
    Ok(())
}

pub(super) fn validate_commit_events<E, Data, Scope>(
    commit: &RunCommit<E, Data, Scope>,
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    for event in &commit.events {
        validate_event_run(&commit.run_id, &commit.session_id, event)?;
        if event.seq() <= 0 {
            return Err(MachineError::InvalidRunEvent {
                reason: "event seq must be positive".to_string(),
            });
        }
    }
    match &commit.finish {
        Some(finish) => {
            if finish.run_id != commit.run_id || finish.session_id != commit.session_id {
                return Err(MachineError::InvalidRunEvent {
                    reason: "finish target does not match committed run".to_string(),
                });
            }
            if !finish.status.is_terminal() {
                return Err(MachineError::InvalidRunEvent {
                    reason: "finish_run requires a terminal status".to_string(),
                });
            }
            let Some(last) = commit.events.last() else {
                return Err(MachineError::InvalidRunEvent {
                    reason: "finish commit requires a terminal event".to_string(),
                });
            };
            if !last.is_terminal() {
                return Err(MachineError::InvalidRunEvent {
                    reason: "finish_run requires a terminal event".to_string(),
                });
            }
            if commit.events[..commit.events.len() - 1]
                .iter()
                .any(RunEvent::is_terminal)
            {
                return Err(MachineError::InvalidRunEvent {
                    reason: "only the last commit event may be terminal".to_string(),
                });
            }
        }
        None => {
            if commit.events.iter().any(RunEvent::is_terminal) {
                return Err(MachineError::InvalidRunEvent {
                    reason: "record_event does not accept terminal events".to_string(),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn validate_next_seq<E, Scope, FinishData>(
    run: &MemoryRun<E, Scope, FinishData>,
    event: &E,
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    if event.seq() <= 0 {
        return Err(MachineError::InvalidRunEvent {
            reason: "event seq must be positive".to_string(),
        });
    }
    if let Some(last) = run.events.last()
        && event.seq() <= last.seq()
    {
        return Err(MachineError::InvalidRunEvent {
            reason: "event seq must increase monotonically".to_string(),
        });
    }
    Ok(())
}

pub(super) fn validate_event_sequence<E, Scope, FinishData>(
    run: &MemoryRun<E, Scope, FinishData>,
    events: &[E],
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    let mut last_seq = run.events.last().map(RunEvent::seq);
    for event in events {
        if event.seq() <= 0 {
            return Err(MachineError::InvalidRunEvent {
                reason: "event seq must be positive".to_string(),
            });
        }
        if let Some(last) = last_seq
            && event.seq() <= last
        {
            return Err(MachineError::InvalidRunEvent {
                reason: "event seq must increase monotonically".to_string(),
            });
        }
        last_seq = Some(event.seq());
    }
    Ok(())
}
