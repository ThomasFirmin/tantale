//! Utility modules for various functionalities used across the optimization framework, 
//! including dominance relations, ordering, archives, handling of raw solution data and objective values, and traits for accessing solution information.

pub mod dominate;
pub use dominate::{Dominate, NdArrayDominate, ParetoFront, IntoParetoFront, NonDominatedSorting, IntoNonDominatedSorting};

pub mod orderable;
pub use orderable::Orderable;

pub mod archives;
pub use archives::OrderedArchive;

pub mod xy;
pub use xy::{Xy, XToNdArray, YToNdArray};

pub mod has_trait;
pub use has_trait::{HasX, HasY, HasFidelity, HasStep, HasStepId, HasId, HasSolInfo, HasUncomputed, HasInfo, HasVariables};