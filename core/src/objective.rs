pub mod obj;
pub use obj::{FuncWrapper, Objective, Stepped};

pub mod codomain;
pub use codomain::{
    Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Cost, CostCodomain,
    CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain, Criteria, FidCodomain,
    FidConstCodomain, FidConstMultiCodomain, FidCostCodomain, FidCostConstCodomain,
    FidCostConstMultiCodomain, FidCostMultiCodomain, FidCriteria, FidMultiCodomain, HasEvalState,
    Multi, MultiCodomain, Single, SingleCodomain,EvalState
};

pub mod outcome;
pub use outcome::{Outcome,FidOutcome};
