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
//! assert_eq!(paired.get_obj().get_id(), paired.get_opt().get_id());
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
//! - **[`Step`]** progress via [`HasStep`]
//! - **[`Fidelity`]** level via [`HasFidelity`]
//!
//! This enables progressive evaluation and resource allocation strategies.
//!
//! ## Structured solutions
//!
//! Solutions can be arranged in different way:
//! - [`Lone`](shape::Lone): Single-domain solution
//! - [`Pair`](shape::Pair): Paired Obj/Opt solutions
//! - [`Batch`](batchtype::Batch): Collections of [`SolutionShape`]
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
//! - [`Domain`](crate::Domain) - Defines solution value types
//! - [`Codomain`](crate::Codomain) - Defines evaluation result types
//! - [`id`] - Solution identifier types
//! - [`partial`] - Concrete solution implementations
//! - [`computed`] - Evaluated solution wrappers
//!

use crate::{
    EvalStep, OptInfo, Outcome,
    domain::{Codomain, Domain},
    objective::Step,
};

use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};

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
pub trait SolInfo: Debug + Serialize + for<'a> Deserialize<'a> {}

/// Trait for objects with a unique solution identifier.
///
/// [`HasId`] provides access to a solution's unique [`Id`], which remains constant across
/// domain transformations. Twin solutions (same point in different domains) share the same [`Id`],
/// enabling tracking and correlation throughout the optimization process.
pub trait HasId<SolId: Id> {
    /// Returns the solution's unique identifier.
    fn get_id(&self) -> SolId;

    /// Checks if another object is a twin (shares the same [`Id`]).
    ///
    /// Two solutions are twins if they represent the same point in different domains
    /// or shapes. Twin solutions always have equal [`Id`]s.
    ///
    /// # Parameters
    ///
    /// * `solb` - Another object with an [`Id`] to compare
    ///
    /// # Returns
    ///
    /// `true` if both objects share the same [`Id`], `false` otherwise.
    fn is_twin<Twin: HasId<SolId>>(&self, solb: Twin) -> bool {
        self.get_id() == solb.get_id()
    }
}

/// Trait for objects carrying solution metadata.
///
/// [`HasSolInfo`] provides access to a solution's associated metadata ([`SolInfo`]), which may
/// include iteration numbers, timestamps, or other optimizer-specific information specifically related to the solution.
pub trait HasSolInfo<Info: SolInfo> {
    /// Returns the solution's [`SolInfo`] wrapped in [`Arc`].
    fn get_sinfo(&self) -> Arc<Info>;
}

/// Trait for objects with an associated objective function value.
///
/// [`HasY`] provides access to a solution's evaluation result ([`TypeCodom`](Codomain::TypeCodom)), representing the output of the
/// objective function. This trait is typically implemented by [`Computed`] solutions.
pub trait HasY<Cod: Codomain<Out>, Out: Outcome> {
    /// Returns the objective function value associated with this solution.
    ///
    /// # Returns
    ///
    /// A shared reference to the codomain value (objective function output).
    fn get_y(&self) -> Arc<Cod::TypeCodom>;
}

/// Trait for objects carrying optimizer-specific metadata.
///
/// [`HasInfo`] provides access to optimizer-specific information that may be attached to
/// solutions, distinct from general [`SolInfo`].
pub trait HasInfo<Info: OptInfo> {
    /// Returns the optimizer-specific metadata for this solution.
    fn get_info(&self) -> Arc<Info>;
}

/// Trait for solutions supporting multi-fidelity evaluation tracking.
///
/// [`HasStep`] enables tracking evaluation progress for objective functions that can be
/// evaluated incrementally (multi-fidelity). The evaluation state is represented by a [`Step`],
/// which can indicate pending evaluation, partial progress, completion, or discard status.
///
/// # Evaluation States
///
/// - **[`Pending`](Step::Pending)**: Awaiting evaluation
/// - **[`Partially(n)`](Step::Partially)**: Evaluated to step `n`
/// - **[`Evaluated`](Step::Evaluated)**: Fully evaluated
/// - **[`Discard`](Step::Discard)**: Marked for rejection without full evaluation
/// - **[`Error`](Step::Error)**: Evaluation failed
pub trait HasStep {
    /// Returns the current evaluation [`Step`].
    fn step(&self) -> Step;

    /// Returns the raw internal evaluation [`EvalStep`].
    fn raw_step(&self) -> EvalStep;

    /// Sets the evaluation state directly via a raw [`EvalStep`].
    fn set_step(&mut self, value: EvalStep);

    /// Marks the solution as pending evaluation.
    ///
    /// Sets the state to [`Pending`](Step::Pending), indicating the solution
    /// has not yet been evaluated.
    fn pending(&mut self);

    /// Marks the solution as partially evaluated to a specific step.
    ///
    /// # Parameters
    ///
    /// * `value` - The step number reached (non-negative)
    fn partially(&mut self, value: isize);

    /// Marks the solution for discard.
    ///
    /// Sets the state to [`Discard`](Step::Discard), indicating the solution
    /// should be rejected without further evaluation (e.g., early stopping).
    fn discard(&mut self);

    /// Marks the solution as fully evaluated.
    ///
    /// Sets the state to [`Evaluated`](Step::Evaluated), indicating the solution
    /// has been completely evaluated at the highest fidelity.
    fn evaluated(&mut self);

    /// Marks the solution as having encountered an error during evaluation.
    ///
    /// Sets the state to [`Error`](Step::Error), indicating evaluation failed.
    fn error(&mut self);
}

/// Trait for solutions with an associated fidelity level.
///
/// [`HasFidelity`] tracks the computational budget or resource level at which a solution
/// was or should be evaluated. This is used in multi-fidelity optimization to control
/// evaluation costs and implement progressive evaluation strategies.
///
/// Higher fidelity values generally mean more accurate but more expensive evaluations.
pub trait HasFidelity {
    /// Returns the current fidelity level.
    fn fidelity(&self) -> Fidelity;

    /// Sets the fidelity level for this solution.
    fn set_fidelity(&mut self, fidelity: Fidelity);
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
/// - [`BasePartial`](partial::BasePartial) - Standard solution implementation
/// - [`FidBasePartial`](partial::FidBasePartial) - With fidelity support
/// - [`Computed`] - Wrapper for evaluated solutions
pub trait Solution<SolId, Dom, SInfo>: HasId<SolId> + HasSolInfo<SInfo>
where
    Self: Sized + Debug + Serialize + for<'de> Deserialize<'de>,
    SolId: Id,
    SInfo: SolInfo,
    Dom: Domain,
{
    /// The native type representing variable values in this domain.
    ///
    /// This type is determined by the [TypeDom](Domain::TypeDom) from the solution's [`Domain`].
    type Raw: Debug + Serialize + for<'de> Deserialize<'de>;

    /// Twin solutions share the same [`Id`] but exist in different domains,
    /// enabling transformations between Obj and Opt representations.
    type Twin<B: Domain>: Solution<SolId, B, SInfo, Twin<Dom> = Self>;

    /// Creates a twin solution in a different domain with the same [`Id`].
    ///
    /// This method constructs a solution in domain `B` that shares the same identifier
    /// as `self`, establishing a twin relationship. This is used internally by
    /// [`Searchspace`](crate::Searchspace) when mapping between Obj and Opt domains.
    ///
    /// Twins are usually wrapped within a [`SolutionShape`] to facilitate access and manipulation of both representation.
    fn twin<B: Domain>(
        &self,
        x: <Self::Twin<B> as Solution<SolId, B, SInfo>>::Raw,
    ) -> Self::Twin<B>;

    /// Returns the raw values for this solution.
    ///
    /// This is the actual sampled point from the domain, represented in the
    /// domain's [`Raw`](Solution::Raw) type.
    fn get_x(&self) -> Self::Raw;
}

/// Trait for objects containing an uncomputed solution.
///
/// [`HasUncomputed`] provides access to the underlying [`Uncomputed`] solution within
/// wrapper types like [`Computed`] or [`SolutionShape`].
pub trait HasUncomputed<SolId: Id, Dom: Domain, SInfo: SolInfo> {
    /// The uncomputed solution type contained within.
    type Uncomputed: Uncomputed<SolId, Dom, SInfo>;

    /// Returns a reference to the contained uncomputed solution.
    fn get_uncomputed(&self) -> &Self::Uncomputed;
}

/// Trait for solutions that have not yet been evaluated.
///
/// [`Uncomputed`] represents solutions generated by sampling or optimization but not yet
/// evaluated by the objective function. These solutions can be converted to [`Computed`]
/// solutions via the [`IntoComputed`] trait and a [`TypeCodom`](Codomain::TypeCodom).
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
pub trait Uncomputed<SolId, Dom, SInfo>: Solution<SolId, Dom, SInfo> + IntoComputed
where
    SolId: Id,
    Dom: Domain,
    SInfo: SolInfo,
{
    /// The uncomputed twin type in an alternative domain.
    ///
    /// Unlike [`Twin`](Solution::Twin), this always produces an [`Uncomputed`] solution,
    /// preserving the uncomputed status across domain transformations.
    type TwinUC<B: Domain>: Uncomputed<SolId, B, SInfo, Twin<Dom> = Self, TwinUC<Dom> = Self>;

    /// Creates an uncomputed twin solution in a different domain.
    ///
    /// # Type Parameters
    ///
    /// * `B` - The target domain
    ///
    /// # Parameters
    ///
    /// * `x` - Variable values in the target domain
    ///
    /// # Returns
    ///
    /// An uncomputed solution in domain `B` with the same [`Id`] as `self`.
    fn twin<B: Domain>(
        &self,
        x: <Self::TwinUC<B> as Solution<SolId, B, SInfo>>::Raw,
    ) -> Self::TwinUC<B>;

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
    fn new<T>(id: SolId, x: T, info: Arc<SInfo>) -> Self
    where
        T: Into<Self::Raw>;

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
    fn default(info: Arc<SInfo>, size: usize) -> Self;

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
    fn default_vec(info: Arc<SInfo>, size: usize, vsize: usize) -> Vec<Self>;
}

/// Trait for converting uncomputed solutions to computed solutions with objective values.
///
/// [`IntoComputed`] enables the transformation from unevaluated solutions to evaluated ones
/// by associating an objective function output ([`TypeCodom`](Codomain::TypeCodom)). This trait
/// also provides the reverse operation to extract the original solution and its value.
///
/// # Usage
///
/// ```text
/// Uncomputed --[into_computed]--> Computed
///                                     |
///                                     v
///                            (Uncomputed, Y) --[extract]
/// ```
///
/// # Typ[`Codomain`]e-Level Design
///
/// The [`Computed`](IntoComputed::Computed) associated type is generic over the codomain,
/// allowing a single uncomputed solution to be converted into computed forms with different
/// outcome types (single-objective, multi-objective, constrained, etc.).
pub trait IntoComputed: Sized {
    /// The computed type wrapping `Self` with an objective value.
    ///
    /// This type must implement [`HasY`] to provide access to the objective function output.
    type Computed<Cod: Codomain<Out>, Out: Outcome>: HasY<Cod, Out>;

    /// Converts this uncomputed solution into a computed one with an objective value.
    ///
    /// # Type Parameters
    ///
    /// * `Cod` - The [`Codomain`] defining the output structure
    /// * `Out` - The outcome type from the objective function
    ///
    /// # Parameters
    ///
    /// * `y` - The objective function output to associate with this solution
    ///
    /// # Returns
    ///
    /// A computed solution containing both `self` and `y`.
    fn into_computed<Cod: Codomain<Out>, Out: Outcome>(
        self,
        y: Arc<Cod::TypeCodom>,
    ) -> Self::Computed<Cod, Out>;

    /// Extracts the uncomputed solution and objective value from a computed solution.
    ///
    /// This is the inverse operation of [`into_computed`](IntoComputed::into_computed),
    /// decomposing a computed solution back into its constituent parts.
    ///
    /// # Type Parameters
    ///
    /// * `Cod` - The [`Codomain`] of the objective value
    /// * `Out` - The [`Outcome`] type
    ///
    /// # Parameters
    ///
    /// * `comp` - The computed solution to decompose
    ///
    /// # Returns
    ///
    /// A tuple of `(uncomputed_solution, objective_value)`.
    fn extract<Cod: Codomain<Out>, Out: Outcome>(
        comp: Self::Computed<Cod, Out>,
    ) -> (Self, Arc<Cod::TypeCodom>);
}

/// Type alias for the twin solution type in an alternative domain.
///
/// [`SolTwin`] extracts the [`Twin`](Solution::Twin) associated type from a solution,
/// providing a convenient way to refer to the solution type in domain `B` that is
/// twin to a solution of type `S` in domain `A`.
pub type SolTwin<S, SolId, A, B, SInfo> = <S as Solution<SolId, A, SInfo>>::Twin<B>;

/// Type alias for the raw value type of a solution.
///
/// [`SolRaw`] extracts the [`Raw`](Solution::Raw) associated type from a solution,
/// providing the native type used to represent variable values in the given domain.
pub type SolRaw<S, SolId, Dom, Info> = <S as Solution<SolId, Dom, Info>>::Raw;

pub mod id;
pub use id::{Id, ParSId, SId};

pub mod partial;
pub use partial::{BaseSol, FidelitySol, Fidelity};

pub mod computed;
pub use computed::Computed;

pub mod batchtype;
pub use batchtype::{Batch, OutBatch};

pub mod shape;
pub use shape::{CompLone, Lone, Pair, SolutionShape};
