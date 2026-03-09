//! # Domain
//!
//! This module defines the core abstractions that describe the **domain** of a
//! variable. A domain specifies what values a variable can take and how to sample valid values.
//!
//! ## Overview
//!
//! The central abstraction is the [`Domain`] trait. A domain:
//! - Defines an associated value, an element from the domain, via [`Domain::TypeDom`]
//! - Provides a sampling method via [`Domain::sample`]
//! - Validates membership via [`Domain::is_in`]
//!
//! Domains are used by [`Var`](crate::variable::var::Var) inside a [`Searchspace`](crate::searchspace::Searchspace)
//! to describe the inputs expected by an [`Objective`](crate::objective::Objective) and explored by an
//! [`Optimizer`](crate::optimizer::Optimizer). The resulting [`Solution`](crate::solution::Solution) contains
//! values whose types are determined by the corresponding domain.
//!
//! ## Type Relationship
//!
//! Each domain has an associated type [`Domain::TypeDom`]. For convenience, the alias
//! [`TypeDom`] maps a domain type to its value type:
//!
//! ## Built-in Domains
//!
//! This module exposes several commonly used domain types:
//! - [`Real`], [`Int`], [`Nat`] and [`Bounded`] for numeric ranges
//! - [`Bool`] for binary values
//! - [`Cat`] for categorical variables
//! - [`Unit`] and [`NoDomain`] for degenerate or placeholder domains
//!
//! Some specific traits such as:
//! - [`Mixed`] for heterogeneous collections of sub-domains
//! - [`Onto`] for mapping an element from one domain to another domain
//!
//! The [`codomain`](crate::domain::codomain) submodule provides codomain abstractions such as
//! [`SingleCodomain`] and [`MultiCodomain`] used for objective outputs.
//!
//! ## Notes
//!
//! Combination of [`Domain`]s allows defining a [`Searchspace`](crate::searchspace::Searchspace).
//! For example:
//! $\mathcal{S} := [`Domain`]_1 \times \cdots \times [`Domain`]_d $
//!
//! ## Macro Integration
//!
//! Domains are constructed directly or through macros like [`objective!`](../../../tantale/macros/macro.objective.html)
//! and [`sp!`](../../../tantale/macros/macro.sp.html). For compatibility with those macros, domain types should
//! provide a `new(...) -> Self` constructor.

#[cfg(doc)]
use crate::objective::Objective;
#[cfg(doc)]
use crate::optimizer::Optimizer;
#[cfg(doc)]
use crate::solution::Solution;
#[cfg(doc)]
use crate::variable::var::Var;

use rand::prelude::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

/// Marker trait for domain types requiring [`Debug`] implementation.
///
/// [`PreDomain`] serves as a prerequisite for [`Domain`], ensuring that all domain types
/// can be formatted for debugging. This is automatically satisfied by deriving [`Debug`] on
/// domain structs.
///
/// # Usage
///
/// This trait is typically not implemented directly by users. But created by
/// the [`objective!`](../../../tantale/macros/macro.objective.html) and
///  [`sp!`](../../../tantale/macros/macro.sp.html) macros when defining domains for objectives and searchspaces.
pub trait PreDomain: Debug {}

/// Core trait defining a domain and its associated value type.
///
/// A [`Domain`] describes a set of valid values that a variable can take during optimization.
/// It provides the essential operations needed by Tantale's type system: determining the Rust type
/// of values in the domain ([`TypeDom`](Domain::TypeDom)), sampling random values...
///
/// # Trait Bounds
///
/// [`Domain`] requires several trait bounds that ensure domains are well-behaved:
/// - [`PreDomain`] - Ensures debugging support
/// - [`Sized`] - Domains must have a known size at compile time
/// - [`PartialEq`] - Domains can be compared for equality
/// - [`Debug`] - Domains can be formatted for debugging output
///
/// # Associated Type: `TypeDom`
///
/// The associated type [`TypeDom`](Domain::TypeDom)  defines what
/// Rust type represents a single value from the [`Domain`].
/// # Required Methods
///
/// Implementors must provide:
/// - [`sample`](Domain::sample) - Generate a random valid value from the domain
/// - [`is_in`](Domain::is_in) - Check whether a value belongs to the domain
///
/// # Constructor Convention
///
/// By convention, domain types should provide a `new(...) -> Self` constructor. This is required
/// for compatibility with Tantale's procedural macros:
/// - [`objective!`](../../../tantale/macros/macro.objective.html) - Defines objective functions with domain constraints
/// - [`sp!`](../../../tantale/macros/macro.sp.html) - Constructs searchspaces from domain specifications
///
/// # Example
///
/// ```ignore
/// use tantale_core::domain::{Domain, PreDomain};
/// use rand::Rng;
///
/// #[derive(Debug, PartialEq)]
/// pub struct BinaryDomain;
///
/// impl PreDomain for BinaryDomain {}
///
/// impl Domain for BinaryDomain {
///     type TypeDom = bool;
///
///     fn sample<R: Rng>(&self, rng: &mut R) -> bool {
///         rng.gen_bool(0.5)
///     }
///
///     fn is_in(&self, _point: &bool) -> bool {
///         true  // All booleans are valid
///     }
/// }
///
/// impl BinaryDomain {
///     pub fn new() -> Self {
///         BinaryDomain
///     }
/// }
/// ```
///
/// # Built-in Implementations
///
/// Tantale provides several built-in domain types:
/// - [`Real`], [`Int`], [`Nat`] - Numeric ranges with bounds
/// - [`Bool`] - Boolean domain
/// - [`Cat`] - Categorical values
/// - [`Mixed`] - Heterogeneous tuple of domains
/// - [`Onto`] - Mapped domains
///
/// # See Also
///
/// - [`TypeDom`] - Type alias for extracting the value type from a domain
/// - [`Var`](crate::variable::var::Var) - Variables that use domains
/// - [`Searchspace`](crate::searchspace::Searchspace) - Collections of domains defining the optimization space
pub trait Domain: PreDomain + Sized + PartialEq + Debug {
    /// The type representing values from this domain.
    ///
    /// [`TypeDom`](Domain::TypeDom) is the associated type that defines what type
    /// represents a single element sampled from the domain.
    ///
    /// # Examples
    ///
    /// For built-in domains:
    /// - [`Real::TypeDom`](crate::domain::Real)` = f64` - Real-valued domains use 64-bit floats
    /// - [`Int::TypeDom`](crate::domain::Int)` = i64` - Integer domains use 64-bit signed integers
    /// - [`Bool::TypeDom`](crate::domain::Bool)` = bool` - Boolean domains use the `bool` type
    /// - [`Cat::TypeDom`](crate::domain::Cat)` = usize` - Categorical domains use indices
    ///
    /// # Usage in [`Solutions`](crate::solution::Solution)
    ///
    /// The [`TypeDom`] determines the type of values stored in [`Solution`](crate::solution::Solution)
    /// instances.
    type TypeDom: Sized
        + PartialEq
        + Clone
        + Display
        + Debug
        + Default
        + Serialize
        + for<'a> Deserialize<'a>;

    /// Generates a random value from the domain.
    ///
    /// This method samples a value from
    /// the domain using the provided random number generator.
    /// See [`Sampler`](crate::sampler::Sampler) for more advanced sampling strategies.
    ///
    /// # Parameters
    ///
    /// * `rng` - A mutable reference to a random number generator implementing [`Rng`](rand::prelude::Rng)
    ///
    /// # Returns
    ///
    /// A randomly sampled value of type [`TypeDom`](Domain::TypeDom) that satisfies [`is_in`](Domain::is_in).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tantale_core::domain::{Domain, Real};
    /// use rand::thread_rng;
    ///
    /// let domain = Real::new(0.0, 1.0);
    /// let mut rng = thread_rng();
    /// let value = domain.sample(&mut rng);
    /// assert!(domain.is_in(&value));
    /// ```
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::TypeDom;
    /// Checks whether a value belongs to the domain.
    /// # Parameters
    ///
    /// * `point` - A reference to a value of type [`TypeDom`](Domain::TypeDom) to validate
    fn is_in(&self, point: &Self::TypeDom) -> bool;
}

/// Type alias for extracting the value type from a domain.
///
/// [`TypeDom<T>`] is a convenience type alias that maps a domain type `T` to its associated
/// [`Domain::TypeDom`] type. This simplifies type signatures and makes code more readable.
///
/// # See Also
///
/// - [`Domain::TypeDom`] - The associated type being aliased
/// - [`Domain`] - The trait defining domains
pub type TypeDom<T> = <T as Domain>::TypeDom;

pub mod nodomain;
pub use nodomain::NoDomain;

pub mod bounded;
pub use bounded::{Bounded, Int, Nat, Real};

pub mod unit;
pub use unit::Unit;

pub mod bool;
pub use bool::Bool;

pub mod cat;
pub use cat::Cat;

pub mod mixed;
pub use mixed::{Mixed, MixedTypeDom};

pub mod onto;
pub use onto::{Linked, LinkObj, LinkOpt, LinkTyObj, LinkTyOpt, Onto, OntoDom};

pub mod codomain;
pub use codomain::{
    Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Cost, CostCodomain,
    CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain, Criteria, FidCriteria,
    Multi, MultiCodomain, Single, SingleCodomain, Accumulator, BestComputed, ParetoComputed,
};
