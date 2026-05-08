use crate::error::MachineError;
use crate::run::RunContext;
use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::fmt::Debug;

pub trait MachineState: Clone + Send + Sync + 'static {
    fn to_json(&self) -> Result<Value, MachineError>;
    fn from_json(value: &Value) -> Result<Self, MachineError>;
}

impl<T> MachineState for T
where
    T: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    fn to_json(&self) -> Result<Value, MachineError> {
        serde_json::to_value(self).map_err(MachineError::Serialization)
    }

    fn from_json(value: &Value) -> Result<Self, MachineError> {
        serde_json::from_value(value.clone()).map_err(MachineError::Deserialization)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transition<Step, Interrupt, Output> {
    Next(Step),
    Interrupt(Interrupt),
    Complete(Output),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeAction<Step> {
    ReenterInterruptedStep,
    JumpTo(Step),
}

#[async_trait]
pub trait Machine: Send + Sync + 'static {
    type Step: Clone + Debug + Serialize + DeserializeOwned + Send + Sync + 'static;
    type State: MachineState;
    type Input: Clone + Send + Sync + 'static;
    type Signal: Send + Sync + 'static;
    type Output: Send + Sync + 'static;
    type Interrupt: Clone + Serialize + DeserializeOwned + Send + Sync + 'static;

    fn start_step(&self) -> Self::Step;

    fn resume_action(&self, _interrupt: &Self::Interrupt) -> ResumeAction<Self::Step> {
        ResumeAction::ReenterInterruptedStep
    }

    fn new_state(
        &self,
        input: &Self::Input,
        previous: Option<&Self::State>,
        snapshot: Option<&Value>,
    ) -> Result<Self::State, MachineError>;

    fn apply_resume_input(
        &self,
        state: &mut Self::State,
        input: &Self::Input,
    ) -> Result<(), MachineError>;

    async fn transition(
        &self,
        step: Self::Step,
        state: &mut Self::State,
        ctx: &RunContext<Self::Input, Self::Step, Self::Signal, Self::Output, Self::Interrupt>,
    ) -> Result<Transition<Self::Step, Self::Interrupt, Self::Output>, MachineError>;
}
