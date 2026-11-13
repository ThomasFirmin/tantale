//! Candidate solution types and relationships.
//!
//! A solution stores a candidate's domain-specific values, identifier, and
//! metadata. [`BaseSol`] represents an unevaluated candidate, while
//! [`Computed`] pairs one with the codomain value produced by evaluation.
//!
//! [`SolutionShape`] links objective-facing and optimizer-facing solutions
//! that represent the same candidate and share an identifier. [`Pair`] stores
//! both representations; [`Lone`] is used when only one side is needed.
//!
//! For multi-fidelity evaluation, [`FidelitySol`] additionally stores a
//! fidelity budget and evaluation [`Step`](crate::objective::Step). The
//! [`Searchspace`](crate::Searchspace) API creates, samples, and maps
//! solutions. See the crate-level documentation for complete, tested usage
//! examples.

use crate::{
    HasId, HasInfo, HasX, Outcome,
    aliases::TypeCodom,
    domain::Domain,
    utils::record::{Recordable, Schema},
};

use serde::{Deserialize, Serialize};
use std::{any::Any, fmt::Debug, sync::Arc};

/// Metadata associated with a solution during optimization.
///
/// [`SolInfo`] provides a way to attach auxiliary information to solutions, such as iteration
/// numbers, timestamps, or optimizer-specific data. This information persists across checkpointing
/// and is available to recorders.
///
/// # Associated Derive Macro
///
/// The `SolInfo` derive macro automatically implements the trait for any struct
/// satisfying the required trait bounds.
pub trait SolInfo:
    Clone
    + Debug
    + Default
    + Serialize
    + Send
    + Sync
    + 'static
    + for<'a> Deserialize<'a>
    + Schema<(), Parts = ()>
    + Recordable<(), Parts = ()>
{
    /// Returns the first sub-info of type `T` found in this info tree, sharing the underlying [`Arc`].
    fn find_info<T: SolInfo>(self: Arc<Self>) -> Option<Arc<T>> {
        (self as Arc<dyn Any + Send + Sync>).downcast::<T>().ok()
    }
}

/// Info returned by a sampler alongside a solution.
#[derive(Serialize, Deserialize)]
pub enum Sampled<I: Debug> {
    /// New info, converted into `P::Info` by the top-level `sample`.
    Fresh(Arc<I>),
    /// The shape already carries its final info; only valid for a shape that was sampled before.
    Reused,
}

impl<I: Debug> Debug for Sampled<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sampled::Fresh(info) => write!(f, "Sampled::Fresh({:?})", info),
            Sampled::Reused => write!(f, "Sampled::Reused"),
        }
    }
}

impl<I: Debug> Clone for Sampled<I> {
    fn clone(&self) -> Self {
        match self {
            Sampled::Fresh(info) => Sampled::Fresh(info.clone()),
            Sampled::Reused => Sampled::Reused,
        }
    }
}

impl<I: Debug> From<I> for Sampled<I> {
    fn from(info: I) -> Self {
        Sampled::Fresh(Arc::new(info))
    }
}

impl<I: SolInfo> Sampled<I> {
    /// Returns the info, recovering it from the shape's current info when it was [`Reused`](Sampled::Reused).
    pub fn resolve<S: SolInfo>(self, shape_info: Arc<S>) -> Arc<I> {
        match self {
            Sampled::Fresh(info) => info,
            Sampled::Reused => shape_info.find_info::<I>().unwrap_or_default(),
        }
    }
}

/// Core trait representing a candidate solution.
///
/// A solution identifies its [`Domain`] and metadata type. The associated raw
/// values are provided by [`HasX`], and the identifier is provided by
/// [`HasId`].
pub trait Solution:
    Debug + Send + Sync + Clone + HasX + HasId + HasInfo + Serialize + for<'de> Deserialize<'de>
{
    /// The [domain](Domain) defining the variable types and constraints for this solution.
    type Dom: Domain;
    /// The metadata type associated with this solution, typically a tuple of [`SolId`] and [`SolInfo`].
    type MetaData: Clone + Default + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

    fn meta(&self) -> Self::MetaData;
}

/// Trait for solutions that have not yet been evaluated.
///
/// Sampling and searchspace operations create uncomputed solutions. They can
/// be paired with an objective codomain value using [`Uncomputed::into_computed`].
/// The constructors also support default-valued placeholders and batches of
/// placeholders.
pub trait Uncomputed: Solution {
    /// Creates a new uncomputed solution with specified values.
    ///
    /// # Parameters
    ///
    /// * `id` - Unique identifier for this solution
    /// * `x` - Variable values convertible to the solution raw-value type
    /// * `info` - Solution metadata
    ///
    /// # Returns
    ///
    /// A new uncomputed solution.
    fn new<T>(meta: Self::MetaData, x: T) -> Self
    where
        T: Into<Self::X>;

    /// Creates a placeholder uncomputed solution with default/zero values.
    ///
    /// Useful for initializing data structures before filling them with actual solutions.
    ///
    /// # Parameters
    ///
    /// * `info` - Solution metadata
    /// * `size` - Number of variables (dimension)
    ///
    /// # Returns
    ///
    /// An uncomputed solution with default values.
    fn default(info: Arc<Self::Info>, size: usize) -> Self;

    /// Creates a vector of placeholder uncomputed solutions.
    ///
    /// Batch version of [`default`](Uncomputed::default) for initializing collections.
    ///
    /// # Parameters
    ///
    /// * `info` - Shared solution metadata
    /// * `size` - Number of variables per solution (dimension)
    /// * `vsize` - Number of solutions to create
    ///
    /// # Returns
    ///
    /// A vector of `vsize` uncomputed solutions, each with `size` default values.
    fn default_vec(info: Arc<Self::Info>, size: usize, vsize: usize) -> Vec<Self>;
    /// Converts this uncomputed solution into a computed one with an objective value.
    ///
    /// # Type Parameters
    ///
    /// * `Cod` - The [`crate::Codomain`] defining the output structure
    /// * `Out` - The outcome type from the objective function
    ///
    /// # Parameters
    ///
    /// * `y` - The objective function output to associate with this solution
    ///
    /// # Returns
    ///
    /// A computed solution containing both `self` and `y`.
    fn into_computed<Out: Outcome>(self, y: Arc<TypeCodom<Out>>) -> Computed<Self, Out>;
}

pub mod id;
pub use id::{ParSId, SId, SolId, StepId, StepSId};

pub mod partial;
pub use partial::{BaseSol, Fidelity, FidelitySol};

pub mod computed;
pub use computed::Computed;

// pub mod batchtype;
// pub use batchtype::{Batch, OutBatch};

pub mod shape;
pub use shape::{Lone, Pair, SolutionShape, IntoComputedShape, UncomputedShape};
