pub mod obj;
pub use obj::{FuncWrapper, Objective, Stepped};

pub mod codomain;
pub use codomain::{
    Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Cost, CostCodomain,
    CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain, Criteria, EvalStep, StepCodomain,
    StepConstCodomain, StepConstMultiCodomain, StepCostCodomain, StepCostConstCodomain,
    StepCostConstMultiCodomain, StepCostMultiCodomain, FidCriteria, StepMultiCodomain, HasEvalStep,
    Multi, MultiCodomain, Single, SingleCodomain,
};

pub mod outcome;
pub use outcome::{FidOutcome, Outcome};
