//! # Solution
//!
//! This module defines the [`Solution`] trait and related types representing candidate solutions
//! in the optimization process. Solutions are points sampled from a [`Searchspace`](crate::Searchspace)
//! and form the fundamental data structure manipulated by optimizers and objective functions.
//!
//! ## Core Concepts
//!
//! ### Dual Domain Solutions
//!
//! Following Tantale's dual domain architecture, there are two types of solutions:
//! - **Obj solutions**: In the objective function's domain
//! - **Opt solutions**: In the optimizer's domain
//!
//! [`Twin`](Solution::twin) [`Solution`] are wrapped within a [`SolutionShape`] to maintain the relationship and bounds between both
//! generated `Obj` and `Opt` [`Solution`]s.
//!
//! Each solution is statically typed by its corresponding [`Domain`].
//!
//! ### Solution Components
//!
//! A [`Solution`] consists of three essential components:
//! 1. **Unique [`Id`]**: Identifies the solution and links [`twin`](Solution::twin) representations across domains
//! 2. **[`Raw`](Solution::Raw) data**: The actual variable values in the domain's native type
//! 3. **[`SolInfo`]**: Metadata associated with the solution (e.g., iteration info)
//!
//! ### Twin Solutions
//!
//! Solutions in different domains that represent the same point share an [`Id`], making them "twins":
//! ```ignore
//! let obj_solution = sp.sample_obj(&mut rng, info.clone());
//! let paired = sp.onto_opt(obj_solution); // Creates twin in Opt domain
//! assert_eq!(paired.get_obj().id(), paired.get_opt().id());
//! ```
//!
//! ## Solution States
//!
//! Solutions progress through different states during optimization:
//!
//! - **[`Uncomputed`]**: Generated but not yet evaluated
//! - **[`Computed`]**: Evaluated with an associated [`Codomain`](crate::Codomain) value
//!
//! The [`IntoComputed`] trait enables conversion from uncomputed to computed states.
//!
//! ## Multi-Fidelity Support
//!
//! For multi-fidelity optimization, solutions can track:
//! - **[`Step`](crate::Step)** progress via [`HasStep`](crate::HasStep)
//! - **[`Fidelity`]** level via [`HasFidelity`](crate::HasFidelity)
//!
//! This enables progressive evaluation and resource allocation strategies.
//!
//! ## Structured solutions
//!
//! Solutions can be arranged in different way:
//! - [`Lone`]: Single-domain solution
//! - [`Pair`]: Paired Obj/Opt solutions
//! - [`Batch`]: Collections of [`SolutionShape`]
//!
//! ## Creation and Usage
//!
//! Solutions are typically created through [`Searchspace`](crate::Searchspace) methods:
//! ```ignore
//! // Sampling creates Uncomputed solutions
//! let solution = sp.sample_opt(&mut rng, info);
//!
//! // Evaluation converts to Computed
//! let outcome = objective_fn(solution.get_x());
//! let computed = solution.into_computed(Arc::new(outcome));
//! ```
//!
//! ## See Also
//!
//! - [`Searchspace`](crate::Searchspace) - Creates and manages solutions
//! - [`Domain`] - Defines solution value types
//! - [`Codomain`](crate::Codomain) - Defines evaluation result types
//! - [`Id`] - Solution identifier types
//! - [`BaseSol`] - Concrete solution implementations
//! - [`Computed`] - Evaluated solution wrappers
//!

use crate::{
    HasId, HasInfo, HasX, Outcome, aliases::TypeCodom, domain::Domain, utils::record::{Recordable, Schema},
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
pub trait SolInfo: Debug + Default + Serialize + Send + Sync + 'static + for<'a> Deserialize<'a> + Schema<(), Parts = ()> + Recordable<(), Parts = ()> {
    /// Returns the first sub-info of type `T` found in this info tree, sharing the underlying [`Arc`].
    fn find_info<T: SolInfo>(self: Arc<Self>) -> Option<Arc<T>> {
        (self as Arc<dyn Any + Send + Sync>).downcast::<T>().ok()
    }
}

/// Info returned by a sampler alongside a solution.
pub enum Sampled<I> {
    /// New info, converted into `P::Info` by the top-level `sample`.
    Fresh(Arc<I>),
    /// The shape already carries its final info; only valid for a shape that was sampled before.
    Reused,
}

impl<I> From<I> for Sampled<I> {
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

/// Core trait representing a candidate solution in the optimization process.
///
/// [`Solution`] is the fundamental abstraction for points in the search space. It combines
/// a unique identifier ([`Id`]), variable values ([`Raw`](Solution::Raw)), and metadata
/// ([`SolInfo`]), with full serialization support for checkpointing.
///
/// # Associated Types
///
/// ## [`Raw`](Solution::Raw)
///
/// The native representation of variable values in this domain. The type depends on the [`Domain`]:
/// - [`Real`](crate::Real): `f64`
/// - [`Nat`](crate::Nat): `i64`
/// - [`Cat`](crate::Cat): `String``
/// - [`Mixed`](crate::Mixed): [`MixedTypeDom`](crate::MixedTypeDom) (heterogeneous values)
///
/// ## [`Twin`](Solution::Twin)
///
/// The solution type in an alternative domain `B`. Twin solutions share the same [`Id`]
/// but have different [`Raw`](Solution::Raw) representations, enabling domain transformations while
/// maintaining solution identity.
///
/// # Concrete Implementations
///
/// - [`BaseSol`] - Standard solution implementation
/// - [`FidelitySol`] - With fidelity support
/// - [`Computed`] - Wrapper for evaluated solutions
pub trait Solution: Debug + Send + Sync + Clone + HasX + HasId + HasInfo + Serialize + for<'de> Deserialize<'de>
{
    /// The [domain](Domain) defining the variable types and constraints for this solution.
    type Dom: Domain;
}

/// Trait for solutions that have not yet been evaluated.
///
/// [`Uncomputed`] represents solutions generated by sampling or optimization but not yet
/// evaluated by the objective function. These solutions can be converted to [`Computed`]
/// solutions via the [`IntoComputed`] trait and a [`TypeCodom`](crate::Codomain::TypeCodom).
///
/// # Type Parameters
///
/// * `SolId` - The unique identifier type
/// * `Dom` - The domain defining variable types and constraints
/// * `SInfo` - The solution metadata type
///
/// # Lifecycle
///
/// ```text
///        Uncomputed --[evaluate]-> Computed
///              ^                       |
///              |                       |
/// TypeCodom <--+------[extract]--------+
/// ```
///
/// # Builder methods
///
/// [`Uncomputed`] provides multiple ways to create solutions:
/// - [`new`](Uncomputed::new) - From explicit values
/// - [`default`](Uncomputed::default) - Placeholder with zero/default values
/// - [`default_vec`](Uncomputed::default_vec) - Batch of placeholders
pub trait Uncomputed: Solution
{
    /// Creates a new uncomputed solution with specified values.
    ///
    /// # Parameters
    ///
    /// * `id` - Unique identifier for this solution
    /// * `x` - Variable values (convertible to [`Raw`](Solution::Raw))
    /// * `info` - Solution metadata
    ///
    /// # Returns
    ///
    /// A new uncomputed solution.
    fn new<T>(id: Self::Id, x: T, info: Arc<Self::Info>) -> Self
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
pub use id::{SolId, ParSId, SId, StepId, StepSId};

pub mod partial;
pub use partial::{BaseSol, Fidelity, FidelitySol};

pub mod computed;
pub use computed::Computed;

// pub mod batchtype;
// pub use batchtype::{Batch, OutBatch};

pub mod shape;
pub use shape::{Lone, Pair, SolutionShape};
