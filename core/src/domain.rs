#[doc(alias = "Domain")]
/// # Domain
/// This crate describes what a domain of a variable is.
/// Most of the domains implements the [`Domain`] type trait [`TypeDom`].
/// It gives the type of a point within this domain.
/// Domains are use in [`crate::core::variable::Variable`] to define the type of the variable,
/// its `TypeObjective` and `TypeOptimizer`, repectively the input type of that variable within
/// the [`crate::core::objective:Objective`] function, and the input type of the
/// [`crate::core::optimizer::Optimizer`].
///
use rand::prelude::ThreadRng;
use std::fmt::{Debug, Display};

/// [`Domain`] is a trait describing the type of a point from the domain it is attached to.
/// It must implement the `default_sampler` and `is_in` methods.
pub trait Domain: Sized + PartialEq {
    type TypeDom: PartialEq + Clone + Copy + Display + Debug;
    /// Default sampling algorithm used to get a random point from
    /// the [`Domain`].
    fn sample(&self, rng: &mut ThreadRng) -> Self::TypeDom;
    /// Returns `true` if a given borrowed `point` is in the domain. Otherwise returns `false`.
    ///
    /// # Parameters
    ///
    /// * `point` : `&`[`Self`]`::`[`TypeDom`](Domain::TypeDom) - a borrowed point from the [`Domain`].
    ///
    fn is_in(&self, point: &Self::TypeDom) -> bool;
}

/// Empty function to statically check [`Onto`] trait between [`Domains`](Domain).
/// It is applied, before they are wrapped into a [`BaseDom`], where relationships can be lost.
/// For example, mapping a [`Cat`] -> [`Cat`], is not relevant (see Notes). And this statically verified
/// property can be lost when two [`Cat`] are wrapped into a [`BaseDom`].
///
/// # Notes
///
/// Some mapping can be irrelevant.
/// If the [`Objective`] requires a [`Cat`], and the [`Optimizer`] can handle [`Cat`],
/// then no mapping should be required.
/// Mapping `["a", "b", "c"]` ([`Objective`]) to `["d", "e", "f"] ([`Optimizer`]),
/// is no relevant. If the [`Optimizer`] can handle a [`Cat`] defined by `["a", "b", "c"]`,
/// it should also be able to handle a [`Cat`] defined by `["d", "e", "f"]`.
/// The same goes for [`Bool`]. There is no need to map a [`Bool`] onto another [`Bool`].
/// Or mapping a [`Cat`] onto a [`Bool`], as defining a mapping between a nominal and bool is not
/// straightforward.
///
pub fn _check_onto<Obj, Opt>(_obj: &Obj, _opt: &Opt)
where
    Obj: Domain + Clone + Display + Debug + Onto<Opt>,
    Opt: Domain + Clone + Display + Debug + Onto<Obj>,
{
}

pub trait Mixed: Domain {}

pub mod bounded;
pub use bounded::{Bounded, DomainBounded, Int, Nat, Real};

pub mod unit;
pub use unit::Unit;

pub mod bool;
pub use bool::Bool;

pub mod cat;
pub use cat::Cat;

pub mod base;
pub use base::{BaseDom, BaseTypeDom};

pub mod onto;
pub use onto::Onto;

pub mod sampler;
pub use sampler::{uniform, uniform_bool, uniform_cat, uniform_int, uniform_nat, uniform_real};

pub mod derrors;
pub use derrors::{DomainBoundariesError, DomainError, DomainOoBError};

pub mod dmacros;
