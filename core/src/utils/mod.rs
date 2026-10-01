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
    HasFidelity, HasId, HasInfo, HasStep, HasStepId, HasUncomputed, HasVariables, HasX,
    HasY, HasState,
};

pub mod wrapped_info;
pub use wrapped_info::WrappedInfo;

pub mod aliases;

pub mod record;

pub mod config;
pub use config::{FolderConfig, SaverConfig};

pub mod communicator;
pub use communicator::Communicator;

use serde::{Deserialize, Serialize};

/// Serializable state of an object.
///
/// Implementations should capture all information required to resume an
/// optimization after checkpointing.
pub trait State
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}