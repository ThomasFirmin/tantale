use crate::recorder::csv::CSVWritable;
use serde::{Deserialize, Serialize};

/// The current state of the evaluation, defined by the user within the [`Outcome`].
/// Associated to [`EvalStep`].
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Step{
    Pending,
    Partially(isize),
    Penultimate,
    Evaluated,
    Error,
    Other(isize),
}

/// The current state of the evaluation, defined by the user within the [`Outcome`].
/// It is not modeled by an enum because of serialization and deserialization to bincode.
/// But is associated to [`Step`].
/// * [`Pending`](Step::Pending) - A unevaluated solution. If [`EvalStep`]$=0$
/// * [`Partially`](Step::Partially) - A not fully evaluated solution. If [`EvalStep`]$>0$
/// * [`Penultimate`](Step::Penultimate) - The second to last step before completed. Used for better ressource management.
///  If [`EvalStep`]$=-1$.
/// * [`Evaluated`](Step::Evaluated) - A fully evaluated solution. If [`EvalStep`]$=-2$.
/// * [`Error`](Step::Error) - A faulty evaluation. If [`EvalStep`]$=-10$
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct EvalStep(pub isize);

impl PartialEq for EvalStep {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl EvalStep {
    pub fn pending() -> Self{
        Self(0)
    }
    pub fn partially(value:usize) -> Self{
        Self(value as isize)
    }
    pub fn penultimate() -> Self{
        Self(-1)
    }
    pub fn completed() -> Self{
        Self(-2)
    }
    pub fn error() -> Self{
        Self(-10)
    }
    pub fn step(&self)->Step{
        if self.0 == 0 {Step::Pending}
        else if self.0 > 0 {Step::Partially(self.0)}
        else if self.0 == -1 {Step::Penultimate}
        else if self.0 == -2 {Step::Evaluated}
        else if self.0 == -10 {Step::Error}
        else {Step::Other(self.0)}
    }
}

impl std::fmt::Display for EvalStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 0 {write!(f, "Pending")}
        else if self.0 == -1 {write!(f, "Penultimate")}
        else if self.0 == -2 {write!(f, "Evaluated")}
        else if self.0 == -10 {write!(f, "Error")}
        else {write!(f, "{}", self.0)}
    }
}

impl CSVWritable<(), ()> for EvalStep {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("step")])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([self.to_string()])
    }
}

pub mod obj;
pub use obj::{FuncWrapper, Objective, Stepped};

pub mod codomain;
pub use codomain::{
    Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Cost, CostCodomain,
    CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain, Criteria, FidCriteria,
    HasEvalStep, Multi, MultiCodomain, Single, SingleCodomain, StepCodomain, StepConstCodomain,
    StepConstMultiCodomain, StepCostCodomain, StepCostConstCodomain, StepCostConstMultiCodomain,
    StepCostMultiCodomain, StepMultiCodomain,
};

pub mod outcome;
pub use outcome::{FidOutcome, Outcome};
