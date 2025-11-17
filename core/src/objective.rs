pub mod obj;
pub use obj::{FuncWrapper, Objective, Stepped};

pub mod codomain;
pub use codomain::{
    Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Cost, CostCodomain,
    CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain, Criteria, EvalStep, FidCriteria,
    HasEvalStep, Multi, MultiCodomain, Single, SingleCodomain, StepCodomain, StepConstCodomain,
    StepConstMultiCodomain, StepCostCodomain, StepCostConstCodomain, StepCostConstMultiCodomain,
    StepCostMultiCodomain, StepMultiCodomain,
};

pub mod outcome;
pub use outcome::{FidOutcome, Outcome};
