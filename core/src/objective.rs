use crate::recorder::csv::CSVWritable;
use serde::{Deserialize, Serialize};

/// The current state of the evaluation, defined by the user within the [`Outcome`].
/// It is not modeled by an [`enum`] because of serialization and deserialization to bincode.
/// * Partially evaluated - A not fully evaluated solution. If $>0$
/// * Completed - A fully evaluated solution. If $=-1$.
/// * Error - A faulty evaluation. If $=-10$
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct EvalStep(pub f64);

impl PartialEq for EvalStep {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl EvalStep {
    pub fn partially(value:f64) -> Self{
        Self(value)
    }
    pub fn completed() -> Self{
        Self(-1.0)
    }
    pub fn error() -> Self{
        Self(-10.0)
    }
    pub fn is_partially(&self) -> bool {
        self.0 > 0.0
    }
    pub fn is_completed(&self) -> bool {
        self.0 == -1.0
    }
    pub fn is_error(&self) -> bool {
        self.0 == -10.0
    }
}

impl std::fmt::Display for EvalStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            -1.0 => write!(f, "Completed"),
            -10.0 => write!(f, "Error"),
            _ => write!(f, "{}", self.0),
        }
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
