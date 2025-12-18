use crate::recorder::csv::CSVWritable;
use serde::{Deserialize, Serialize};

/// The current state of the evaluation, defined by the user within the [`Outcome`].
/// Associated to [`EvalStep`].
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Step{
    Pending,
    Partially(isize),
    Evaluated,
    Discard,
    Error,
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self{
            Step::Pending => write!(f,"Pending"),
            Step::Partially(v) => write!(f,"{}",v),
            Step::Evaluated => write!(f,"Evaluated"),
            Step::Discard => write!(f,"Discarded"),
            Step::Error => write!(f,"Error"),
        }
    }
}

impl CSVWritable<(), ()> for Step {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("step")])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([self.to_string()])
    }
}

/// The current state of the evaluation, defined by the user within the [`Outcome`].
/// It is not modeled by an enum because of serialization and deserialization to bincode.
/// But is associated to [`Step`].
/// * [`Pending`](Step::Pending) - A unevaluated solution. If [`EvalStep`]$=0$
/// * [`Partially`](Step::Partially) - A not fully evaluated solution. If [`EvalStep`]$>0$
/// * [`Evaluated`](Step::Evaluated) - A fully evaluated solution. If [`EvalStep`]$=-2$.
/// * [`Discard`](Step::Discard) - Used by [`Optimizer`](crate::Optimizer) when the evaluation must be interrupted.
///  If [`EvalStep`]$=-9$.
/// * [`Error`](Step::Error) - A faulty evaluation. If [`EvalStep`]$=-10$
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct EvalStep(pub isize);

impl PartialEq for EvalStep {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl From<Step> for EvalStep{
    fn from(value: Step) -> Self {
        match value{
            Step::Pending => Self(0),
            Step::Partially(v) => Self(v),
            Step::Evaluated => Self(-1),
            Step::Discard => Self(-9),
            Step::Error => Self(-10),
        }
    }
}

impl From<EvalStep> for Step{
    fn from(value: EvalStep) -> Self {
        match value{
            EvalStep(0) => Step::Pending,
            EvalStep(-1) => Step::Evaluated,
            EvalStep(-9) => Step::Discard,
            EvalStep(-10) => Step::Error,
            EvalStep(v) => Step::Partially(v),
        }
    }
}

impl EvalStep {
    pub fn pending() -> Self{ Self(0) }
    pub fn partially(value:usize) -> Self{ Self(value as isize) }
    pub fn completed() -> Self{ Self(-1) }
    pub fn discard() -> Self{ Self(-1) }
    pub fn error() -> Self{ Self(-10) }
}

impl std::fmt::Display for EvalStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 0 {write!(f, "Pending")}
        else if self.0 == -1 {write!(f, "Evaluated")}
        else if self.0 == -9 {write!(f, "Discard")}
        else if self.0 == -10 {write!(f, "Error")}
        else {write!(f, "{}", self.0)}
    }
}

pub mod obj;
pub use obj::{FuncWrapper, Objective, Stepped};

pub mod codomain;
pub use codomain::{
    Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Cost, CostCodomain,
    CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain, Criteria, FidCriteria,
    HasEvalStep, Multi, MultiCodomain, Single, SingleCodomain};

pub mod outcome;
pub use outcome::{FidOutcome, Outcome};
