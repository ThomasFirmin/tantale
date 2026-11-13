//! # Searchspace
//!
//! This module provides the [`Searchspace`] trait and related types for defining the optimization
//! search domain. A searchspace specifies the valid variable values that an optimizer can explore
//! and an objective function can evaluate.
//!
//! ## Overview
//!
//! A [`Searchspace`] is typed by:
//! - An **Obj domain**: The domain for the objective function's input
//! - An **Opt domain**: The domain for the optimizer's representation
//!   `Obj` and `Opt` types are related via [`Onto`](crate::Onto)
//!
//! ## Dual Domain Architecture
//!
//! Tantale uses a **dual domain** approach where each variable has two representations:
//!
//! - **Obj (Objective)**: The domain used by the objective function ([`FuncWrapper`]). This is typically the natural
//!   parameter space of the problem (e.g., categorical choices, specific numeric ranges).
//! - **Opt (Optimizer)**: The domain used by the optimizer ([`Sampler`](crate::Sampler)) for search. This may be a continuous
//!   relaxation or transformation suitable for the optimization algorithm.
//!
//! The searchspace handles automatic mapping between these domains, allowing optimizers to work
//! in their preferred representation while objectives receive values in their expected format.
//!
//! ## Construction
//!
//! Searchspaces are typically constructed using macros:
//! - [`hpo!`](../../../doc/tantale_macros/macro.hpo.html) - Creates a standalone searchspace instance
//! - [`objective!`](../../../doc/tantale_macros/macro.objective.html) - Defines a searchspace within the objective function
//!
//!
//! # Example with [`hpo!`](../../../doc/tantale_macros/macro.hpo.html).
//!
//! This is a mock example to illustrate the usage of the searchspace and its mapping functions.
//! See actual examples within the repository for more detailed use cases.
//!
//! See the crate-level documentation for complete, tested usage examples.
//!
//! # Example with [`objective!`](../../../doc/tantale_macros/macro.objective.html).
//!
//! See the crate-level documentation for complete, tested usage examples.
use crate::{
    FuncWrapper, Solution, Uncomputed,
    aliases::{GetRaw, GetSObj, GetSOpt},
    domain::onto::Linked,
    solution::{SolutionShape, shape::IntoComputedShape},
};

use rand::prelude::Rng;

/// Core trait defining the searchspace for optimization problems.
///
/// A [`Searchspace`] manages the relationship between objective and optimizer representations
/// of solutions. It provides methods for sampling, mapping, and validating solutions in both
/// domains. This trait is central to Tantale's type-safe optimization framework.
///
/// # Dual [`Domains`](crate::Domain)
///
/// Each searchspace maintains two domain representations:
/// - **Obj domain** ([`Self::Obj`](Linked::Obj)): Used by the objective function
/// - **Opt domain** ([`Self::Opt`](Linked::Opt)): Used by the optimizer
///
/// The searchspace provides bidirectional mapping between these domains via
/// [`onto_opt`](Searchspace::onto_opt) and [`onto_obj`](Searchspace::onto_obj).
///
/// # Associated Types
///
/// - [`Shape`](Searchspace::Shape) - The paired solution shape containing both Obj and Opt representations.
///   It can be :
///   * a [`Pair`](crate::solution::shape::Pair) when both `Obj` and `Opt` sides are defined
///   * a [`Lone`](crate::solution::shape::Lone) when only the `Obj` side is defined
///
/// # Key Operations
///
/// ## Sampling
/// - [`sample_obj`](Searchspace::sample_obj) - Generate random Obj domain solution
/// - [`sample_opt`](Searchspace::sample_opt) - Generate random Opt domain solution
/// - [`sample_pair`](Searchspace::sample_pair) - Generate paired Obj/Opt solution ([`SolutionShape`])
///
/// ## Mapping
/// - [`onto_opt`](Searchspace::onto_opt) - Map from Obj to Opt domain
/// - [`onto_obj`](Searchspace::onto_obj) - Map from Opt to Obj domain
///
/// ## Validation
/// - [`contains_obj`](Searchspace::contains_obj) - Check if solution is valid in Obj domain
/// - [`contains_opt`](Searchspace::contains_opt) - Check if solution is valid in Opt domain
///
/// # Usage
///
/// See the crate-level documentation for complete, tested usage examples.
///
/// # See Also
///
/// - [`Sp`] - Concrete searchspace implementation
/// - [`Var`](crate::Var) - Individual variable definitions
/// - [`Linked`] - Trait linking Obj and Opt domains
/// - [`hpo!`](../../../tantale/macros/macro.hpo.html) - Macro for creating searchspaces
/// - [`objective!`](../../../tantale/macros/macro.objective.html) - Macro for defining searchspaces and wrap objective functions
pub trait Searchspace<F: FuncWrapper<Dom = Self::Obj>>: Linked + Send + Sync {
    /// The paired solution shape containing the Obj and Opt solution representations.
    type Shape: SolutionShape<
            ObjSol = F::Sol<Self::Obj>,
            OptSol = F::Sol<Self::Opt>,
            Obj = Self::Obj,
            Opt = Self::Opt,
            Info = F::Info,
        > + IntoComputedShape;

    /// Create a new solution from a raw solution in the Obj domain.
    fn new_obj<T: Into<GetRaw<F::Sol<Self::Obj>>>>(&self, x: T) -> Self::Shape {
        let meta = <F::Sol<Self::Obj> as Solution>::MetaData::default();
        let sol = F::Sol::<Self::Obj>::new(meta, x);
        self.onto_opt(sol)
    }

    /// Create a new solution from a raw solution in the Opt domain.
    fn new_opt<T: Into<GetRaw<F::Sol<Self::Opt>>>>(&self, x: T) -> Self::Shape {
        let meta = <F::Sol<Self::Opt> as Solution>::MetaData::default();
        let sol = F::Sol::<Self::Opt>::new(meta, x);
        self.onto_obj(sol)
    }

    /// Maps a solution from the Obj domain to the Opt domain.
    ///
    /// This method transforms an objective function solution into the optimizer's representation
    /// using the [`Onto`](crate::Onto) mappings defined for each variable in the searchspace.
    /// The resulting solution maintains the same [`SolId`](crate::solution::SolId) as the input.
    ///
    /// # Parameters
    ///
    /// * `inp` - Solution in the Obj domain
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn onto_opt(&self, inp: GetSObj<Self::Shape>) -> Self::Shape;
    /// Maps a solution from the Opt domain to the Obj domain.
    ///
    /// # Parameters
    ///
    /// * `inp` - Solution in the Opt domain
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn onto_obj(&self, inp: GetSOpt<Self::Shape>) -> Self::Shape;
    /// Generates a random solution in the Obj (objective function) domain.
    ///
    /// Samples each variable using its Obj domain sampler.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `info` - Shared solution metadata
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn sample_obj<R: Rng>(&self, rng: &mut R) -> GetSObj<Self::Shape> {
        let meta = <F::Sol<Self::Obj> as Solution>::MetaData::default();
        Uncomputed::new(meta, self.sample_raw_obj(rng))
    }
    /// Generates raw solution data in the Obj (objective function) domain.
    ///
    /// Samples each variable using its Obj domain sampler.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `info` - Shared solution metadata
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn sample_raw_obj<R: Rng>(&self, rng: &mut R) -> GetRaw<GetSObj<Self::Shape>>;
    /// Generates a random solution in the Opt (optimizer) domain.
    ///
    /// Samples each variable using its Opt domain sampler.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn sample_opt<R: Rng>(&self, rng: &mut R) -> GetSOpt<Self::Shape> {
        let meta = <F::Sol<Self::Opt> as Solution>::MetaData::default();
        Uncomputed::new(meta, self.sample_raw_opt(rng))
    }
    /// Generates raw solution data in the Opt (optimizer) domain.
    ///
    /// Samples each variable using its Opt domain sampler.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `info` - Shared solution metadata
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn sample_raw_opt<R: Rng>(&self, rng: &mut R) -> GetRaw<GetSOpt<Self::Shape>>;
    /// Generates a random paired solution containing both Obj and Opt representations.
    ///
    /// Samples in the Obj domain and automatically creates the paired Opt representation
    /// via mapping. Both representations share the same [`SolId`](crate::solution::SolId).
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn sample_pair<R: Rng>(&self, rng: &mut R) -> Self::Shape;
    /// Validates that a solution belongs to the Obj domain of this searchspace.
    ///
    /// Checks each variable value against its Obj domain constraints using
    /// [`Domain::contains`](crate::Domain::contains).
    ///
    /// # Parameters
    ///
    /// * `inp` - Solution to validate
    ///
    /// # Returns
    ///
    /// `true` if all variable values satisfy their Obj domain constraints, `false` otherwise.
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn contains_obj(&self, inp: &GetRaw<GetSObj<Self::Shape>>) -> bool;
    /// Validates that a solution belongs to the Opt domain of this searchspace.
    ///
    /// Checks each variable value against its Opt domain constraints using
    /// [`Domain::contains`](crate::Domain::contains).
    ///
    /// # Parameters
    ///
    /// * `inp` - Solution to validate
    ///
    /// # Returns
    ///
    /// `true` if all variable values satisfy their Opt domain constraints, `false` otherwise.
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn contains_opt(&self, inp: &GetRaw<GetSOpt<Self::Shape>>) -> bool;
    /// Maps a [`Solution`] of type `Opt` onto a [`Solution`] of type `Obj`.
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn vec_onto_obj(&self, inp: Vec<GetSOpt<Self::Shape>>) -> Vec<Self::Shape>;
    /// Maps a vector of solutions from the Obj domain to the Opt domain.
    ///
    /// Batch version of [`onto_opt`](Searchspace::onto_opt) for efficient processing of
    /// multiple solutions.
    ///
    /// # Parameters
    ///
    /// * `inp` - Vector of solutions in the Obj domain
    ///
    /// # Returns
    ///
    /// Vector of paired solutions with both Obj and Opt representations.
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn vec_onto_opt(&self, inp: Vec<GetSObj<Self::Shape>>) -> Vec<Self::Shape>;
    /// Generates multiple random solutions in the Obj domain.
    ///
    /// Batch version of [`sample_obj`](Searchspace::sample_obj) for efficient generation
    /// of multiple solutions.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `size` - Number of solutions to generate
    /// * `info` - Shared solution metadata
    ///
    /// # Returns
    ///
    /// Vector of randomly sampled solutions in the Obj domain, each with a unique [`SolId`](crate::solution::SolId).
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn vec_sample_obj<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<GetSObj<Self::Shape>>;
    /// Generates multiple random raw solutions in the Obj domain.
    ///
    /// Batch version of [`sample_raw_obj`](Searchspace::sample_raw_obj) for efficient generation
    /// of multiple solutions.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `size` - Number of solutions to generate
    ///
    /// # Returns
    ///
    /// Vector of randomly sampled solutions in the Obj domain, each with a unique [`SolId`](crate::solution::SolId).
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn vec_sample_raw_obj<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
    ) -> Vec<GetRaw<GetSObj<Self::Shape>>>;
    /// Generates multiple random solutions in the Opt domain.
    ///
    /// Batch version of [`sample_opt`](Searchspace::sample_opt) for efficient generation
    /// of multiple solutions.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `size` - Number of solutions to generate
    /// * `info` - Shared solution metadata
    ///
    /// # Returns
    ///
    /// Vector of randomly sampled solutions in the Opt domain, each with a unique [`SolId`](crate::solution::SolId).
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn vec_sample_opt<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<GetSOpt<Self::Shape>>;
    /// Generates multiple random raw solutions in the Opt domain.
    ///
    /// Batch version of [`sample_raw_obj`](Searchspace::sample_raw_obj) for efficient generation
    /// of multiple solutions.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `size` - Number of solutions to generate
    ///
    /// # Returns
    ///
    /// Vector of randomly sampled solutions in the Opt domain, each with a unique [`SolId`](crate::solution::SolId).
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn vec_sample_raw_opt<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
    ) -> Vec<GetRaw<GetSOpt<Self::Shape>>>;
    /// Generates multiple random [`SolutionShape`] with both Obj and Opt representations.
    ///
    /// Batch version of [`sample_pair`](Searchspace::sample_pair) for efficient generation
    /// of multiple paired solutions.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `size` - Number of paired solutions to generate
    /// * `info` - Shared solution metadata
    ///
    /// # Returns
    ///
    /// Vector of paired solutions ([`SolutionShape`]), each with both Obj and Opt representations sharing a unique [`SolId`](crate::solution::SolId).
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn vec_sample_pair<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<Self::Shape>;

    /// Generates multiple `Obj` solutions and applies a transformation to each.
    ///
    /// This is similar to [`vec_sample_obj`](Searchspace::vec_sample_obj), but allows a
    /// user-provided closure to post-process each sampled `Obj` solution before returning the vector.
    /// This is useful for adding custom metadata, mutating fields, or enforcing additional
    /// constraints on the generated `Obj` solutions.
    ///
    /// # Parameters
    ///
    /// * `f` - Closure applied to each sampled `Obj` solution
    /// * `rng` - Random number generator
    /// * `size` - Number of `Obj` solutions to generate
    /// * `info` - Shared solution metadata
    ///
    /// # Returns
    ///
    /// Vector of `Obj` solutions after applying `f` to each sample.
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn vec_apply_obj<A, R>(&self, f: A, rng: &mut R, size: usize) -> Vec<GetSObj<Self::Shape>>
    where
        A: Fn(GetSObj<Self::Shape>) -> GetSObj<Self::Shape> + Send + Sync,
        R: Rng;
    /// Generates multiple `Opt` solutions and applies a transformation to each.
    ///
    /// This is similar to [`vec_sample_opt`](Searchspace::vec_sample_opt), but allows a
    /// user-provided closure to post-process each sampled `Opt` solution before returning the vector.
    /// This is useful for adding custom metadata, mutating fields, or enforcing additional
    /// constraints on the generated `Opt` solutions.
    ///
    /// # Parameters
    ///
    /// * `f` - Closure applied to each sampled `Opt` solution
    /// * `rng` - Random number generator
    /// * `size` - Number of `Opt` solutions to generate
    /// * `info` - Shared solution metadata
    ///
    /// # Returns
    ///
    /// Vector of `Opt` solutions after applying `f` to each sample.
    ///
    /// See the crate-level documentation for complete, tested usage examples.
    fn vec_apply_opt<A, R>(&self, f: A, rng: &mut R, size: usize) -> Vec<GetSOpt<Self::Shape>>
    where
        A: Fn(GetSOpt<Self::Shape>) -> GetSOpt<Self::Shape> + Send + Sync,
        R: Rng;
}

pub mod spbase;
pub use spbase::Sp;

pub mod sppar;
pub use sppar::SpPar;
