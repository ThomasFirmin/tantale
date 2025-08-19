pub mod obj;
pub use obj::{ObjBase, Objective};

pub mod codomain;
pub use codomain::{
    Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Criteria, FidelCodomain,
    FidelConstCodomain, FidelConstMultiCodomain, FidelMultiCodomain, Fidelity, Multi,
    MultiCodomain, Single, SingleCodomain,
};

pub mod outcome;
pub use outcome::{LinkedOutcome, Outcome};
