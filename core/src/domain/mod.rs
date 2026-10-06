//! # Domain
//!
//! This module defines the core abstractions that describe the **domain** of a
//! variable. A domain specifies what values a variable can take and how to sample valid values.
//!
//! ## Overview
//!
//! The central abstraction is the [`Domain`] trait. A domain:
//! - Defines an associated value, an element from the domain, via [`Domain::Element`]
//! - Provides a sampling method via [`Domain::sample`]
//! - Validates membership via [`Domain::contains`]
//!
//! Domains are used by [`Var`] inside a [`Searchspace`](crate::searchspace::Searchspace)
//! to describe the inputs expected by an [`Objective`] and explored by an
//! [`Optimizer`]. The resulting [`Solution`] contains
//! values whose types are determined by the corresponding domain.
//!
//! ## Type Relationship
//!
//! Each domain has an associated type [`Domain::Element`]. For convenience, the alias
//! [`Element`] maps a domain type to its value type:
//!
//! ## Built-in Domains
//!
//! This module exposes several commonly used domain types:
//! - [`Real`], [`Int`], [`Nat`] and [`Bounded`] for numeric ranges
//! - [`Bool`] for binary values
//! - [`Unit`] and [`NoDomain`] for degenerate or placeholder domains
//! - [`Mixed`] for heterogeneous domains
//! - [`Grid`] for Cartesian products of discrete values made of [`GridDom`]s
//! - [`GridDom`] for defining discrete sets of values
//!   - [`GridReal`] for discrete real values
//!   - [`GridInt`] for discrete integer values
//!   - [`GridNat`] for discrete natural values
//!   - [`Cat`] for discrete categorical values
//!
//!
//!
//! Some specific traits such as:
//! - [`Onto`] for mapping an element from one domain to another domain
//! - [`OntoDom`] for defining [`Domain`]s that are mapped to another domain
//!
//! The [`codomain`] submodule provides codomain abstractions such as
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
//! and [`hpo!`](../../../tantale/macros/macro.hpo.html). For compatibility with those macros, domain types should
//! provide a `new(...) -> Self` constructor.

#[cfg(doc)]
use crate::objective::OneStep;
#[cfg(doc)]
use crate::optimizer::Sampler;
#[cfg(doc)]
use crate::solution::Solution;
#[cfg(doc)]
use crate::variable::var::Var;

use num::Num;
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
///  [`hpo!`](../../../tantale/macros/macro.hpo.html) macros when defining domains for objectives and searchspaces.
pub trait PreDomain: Debug {}

pub trait Element
where 
    Self: Sized
        + PartialEq
        + Clone
        + Display
        + Debug
        + Default
        + Serialize
        + for<'a> Deserialize<'a>
{}

impl<T> Element for T
where 
    T: Sized
        + PartialEq
        + Clone
        + Display
        + Debug
        + Default
        + Serialize
        + for<'a> Deserialize<'a>
{}

/// Core trait defining a domain and its associated value type.
///
/// A [`Domain`] describes a set of valid values that a variable can take during optimization.
/// It provides the essential operations needed by Tantale's type system: determining the Rust type
/// of values in the domain ([`Element`](Domain::Element)), sampling random values...
///
/// # Trait Bounds
///
/// [`Domain`] requires several trait bounds that ensure domains are well-behaved:
/// - [`PreDomain`] - Ensures debugging support
/// - [`Sized`] - Domains must have a known size at compile time
/// - [`PartialEq`] - Domains can be compared for equality
/// - [`Debug`] - Domains can be formatted for debugging output
///
/// # Associated Type: `Element`
///
/// The associated type [`Element`](Domain::Element)  defines what
/// Rust type represents a single value from the [`Domain`].
/// # Required Methods
///
/// Implementors must provide:
/// - [`sample`](Domain::sample) - Generate a random valid value from the domain
/// - [`contains`](Domain::contains) - Check whether a value belongs to the domain
///
/// # Constructor Convention
///
/// By convention, domain types should provide a `new(...) -> Self` constructor. This is required
/// for compatibility with Tantale's procedural macros:
/// - [`objective!`](../../../tantale/macros/macro.objective.html) - Defines objective functions with domain constraints
/// - [`hpo!`](../../../tantale/macros/macro.hpo.html) - Constructs searchspaces from domain specifications
///
/// # Example
///
/// ```
/// use tantale::core::domain::{Domain, PreDomain};
/// use rand::{RngExt, prelude::{IteratorRandom, Rng}};
///
/// #[derive(Debug, PartialEq)]
/// pub struct BinaryDomain;
///
/// impl PreDomain for BinaryDomain {}
///
/// impl Domain for BinaryDomain {
///     type Element = bool;
///
///     fn sample<R: Rng>(&self, rng: &mut R) -> bool {
///         rng.random_bool(0.5)
///     }
///
///     fn contains(&self, _point: &bool) -> bool {
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
/// - [`Element`] - Type alias for extracting the value type from a domain
/// - [`Var`] - Variables that use domains
/// - [`Searchspace`](crate::Searchspace) - Collections of domains defining the optimization space
pub trait Domain:
    PreDomain + Sized + PartialEq + Debug + Serialize + for<'a> Deserialize<'a>
{
    /// The type representing values from this domain.
    ///
    /// [`Element`](Domain::Element) is the associated type that defines what type
    /// represents a single element sampled from the domain.
    ///
    /// # Examples
    ///
    /// For built-in domains:
    /// - [`Real::Element`](Real)` = f64` - Real-valued domains use 64-bit floats
    /// - [`Int::Element`](Int)` = i64` - Integer domains use 64-bit signed integers
    /// - [`Bool::Element`](Bool)` = bool` - Boolean domains use the `bool` type
    /// - [`Cat::Element`](Cat)` = usize` - Categorical domains use indices
    ///
    /// # Usage in [`Solutions`](crate::Solution)
    ///
    /// The [`Element`] determines the type of values stored in [`Solution`](crate::Solution).
    /// instances.
    type Element: Element;

    /// Generates a random value from the domain.
    ///
    /// This method samples a value from
    /// the domain using the provided random number generator.
    /// See [`DomainSampler`](crate::sampler::DomainSampler) for more advanced sampling strategies.
    ///
    /// # Parameters
    ///
    /// * `rng` - A mutable reference to a random number generator implementing [`Rng`]
    ///
    /// # Returns
    ///
    /// A randomly sampled value of type [`Element`](Domain::Element) that satisfies [`contains`](Domain::contains).
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Domain, Real, Uniform};
    ///
    /// let domain = Real::new(0.0, 1.0, Uniform);
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    /// let value = domain.sample(&mut rng);
    /// assert!(domain.contains(&value));
    /// ```
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Element;
    /// Checks whether a value belongs to the domain.
    /// # Parameters
    ///
    /// * `point` - A reference to a value of type [`Element`](Domain::Element) to validate
    fn contains(&self, point: &Self::Element) -> bool;
}

pub trait NumericalDomain: Domain
where
    Self::Element: Num,
{
    /// Retrieves the bounds of the numerical domain.
    fn get_bounds(&self) -> (Self::Element, Self::Element);

    /// Retrieves references to the bounds of the numerical domain.
    fn get_ref_bounds(&self) -> (&Self::Element, &Self::Element);
}

pub trait CategoricalDomain: Domain
where
    Self::Element: PartialEq,
{
    /// Retrieves the list of features in the categorical domain.
    fn get_features(&self) -> &[Self::Element];
    /// Retrieves the size of the categorical domain, i.e., the number of distinct categories.
    fn size(&self) -> usize;
}

/// Type alias for extracting the value type from a domain.
///
/// [`Elem<T>`] is a convenience type alias that maps a domain type `T` to its associated
/// [`Domain::Element`] type. This simplifies type signatures and makes code more readable.
///
/// # See Also
///
/// - [`Domain::Element`] - The associated type being aliased
/// - [`Domain`] - The trait defining domains
pub type Elem<T> = <T as Domain>::Element;

pub mod nodomain;
pub use nodomain::NoDomain;

pub mod bounded;
pub use bounded::{Bounded, Int, Nat, Real};

pub mod unit;
pub use unit::Unit;

pub mod bool;
pub use bool::Bool;

pub mod grid;
pub use grid::{Cat, Grid, GridDom, GridInt, GridNat, GridReal};

pub mod mixed;
pub use mixed::{Mixed, MixedTypeDom};

pub mod onto;
pub use onto::{Linked, Onto, OntoDom};

pub mod codomain;
pub use codomain::{
    Accumulator, BestAccumulator, Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Cost,
    CostCodomain, CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain,
    Multi, MultiCodomain, ParetoAccumulator, Single, SingleCodomain,
    ElemSingleCodomain,
    ElemCostCodomain,
    ElemConstCodomain,
    ElemCostConstCodomain,
    ElemMultiCodomain,
    ElemCostMultiCodomain,
    ElemConstMultiCodomain,
    ElemCostConstMultiCodomain,
};