//! Utility modules for various functionalities used across the optimization framework,
//! including dominance relations, ordering, archives, handling of raw solution data and objective values, and traits for accessing solution information.

pub mod dominate;
use std::fmt::Debug;

pub use dominate::{
    Dominate, DominateView, IntoNonDominatedSorting, IntoParetoFront, NdArrayDominate,
    NonDominatedSorting, ParetoFront,
};

pub mod orderable;
pub use orderable::Orderable;

pub mod archives;
pub use archives::OrderedArchive;

pub mod xy;
pub use xy::{AsX, XToNdArray, Xy, YToNdArray};

pub mod has_trait;
pub use has_trait::{
    HasFidelity, HasId, HasInfo, HasState, HasStep, HasStepId, HasUncomputed, HasVariables, HasX,
    HasY,
};

pub mod wrapped_info;
pub use wrapped_info::WrappedInfo;

pub mod aliases;

pub mod record;
pub use record::{Recordable, Schema};

pub mod config;
pub use config::FolderConfig;

pub mod communicator;
pub use communicator::Communicator;

use serde::{Deserialize, Serialize};

use crate::{Sampled, SolInfo, SolutionShape};

/// Serializable state of an object.
///
/// Implementations should capture all information required to resume an
/// optimization after checkpointing.
pub trait State
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}

/// A wrapper struct that holds a solution shape and its associated [`Sampled`] info.
#[derive(Serialize, Deserialize)]
#[serde(bound = "S: SolutionShape, I: SolInfo")]
pub struct ShapeWithInfo<S: SolutionShape, I: SolInfo>(pub S, pub Sampled<I>);

impl<S: SolutionShape, I: SolInfo> Debug for ShapeWithInfo<S, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShapeWithInfo")
            .field("shape", &self.0)
            .field("info", &self.1)
            .finish()
    }
}

impl<S: SolutionShape, I: SolInfo> Clone for ShapeWithInfo<S, I> {
    fn clone(&self) -> Self {
        ShapeWithInfo(self.0.clone(), self.1.clone())
    }
}

impl<S: SolutionShape, I: SolInfo> AsRef<S> for ShapeWithInfo<S, I> {
    fn as_ref(&self) -> &S {
        &self.0
    }
}

impl<S: SolutionShape, I: SolInfo> AsMut<S> for ShapeWithInfo<S, I> {
    fn as_mut(&mut self) -> &mut S {
        &mut self.0
    }
}

impl<S: SolutionShape, I: SolInfo> From<(S, I)> for ShapeWithInfo<S, I> {
    fn from(pair: (S, I)) -> Self {
        ShapeWithInfo(pair.0, Sampled::Fresh(pair.1.into()))
    }
}

impl<S: SolutionShape, I: SolInfo> From<ShapeWithInfo<S, I>> for (S, Sampled<I>) {
    fn from(val: ShapeWithInfo<S, I>) -> Self {
        (val.0, val.1)
    }
}
