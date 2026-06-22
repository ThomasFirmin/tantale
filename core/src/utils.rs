//! Utility modules for various functionalities used across the optimization framework,
//! including dominance relations, ordering, archives, handling of raw solution data and objective values, and traits for accessing solution information.

pub mod dominate;
pub use dominate::{
    Dominate, IntoNonDominatedSorting, IntoParetoFront, NdArrayDominate, NonDominatedSorting,
    ParetoFront,
};

pub mod orderable;
pub use orderable::Orderable;

pub mod archives;
pub use archives::OrderedArchive;

pub mod xy;
pub use xy::{XToNdArray, Xy, YToNdArray};

pub mod has_trait;
pub use has_trait::{
    HasFidelity, HasId, HasInfo, HasSolInfo, HasStep, HasStepId, HasUncomputed, HasVariables, HasX,
    HasY,
};
