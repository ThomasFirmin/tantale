//! # Domain
//! This module describes what a domain of a variable is.
//! Most of the domains implements the [`Domain`] type trait [`TypeDom`](Domain::TypeDom).
//! It defines the type of a point sampled within this [`Domain`].
//! [`Domains`](Domain) are used in [`Variable`](crate::variable::var::Var) to define the type of the variable for the
//! [`Objective`](crate::objective::Objective) and the [`Optimizer`](crate::optimizer::Optimizer)
//! [`Solutions`](crate::solution::Solution).
//! Each [`Domain`] has an associated type [`TypeDom`](Domain::TypeDom), allowing to define the type of
//! a single element from a [`Solution`](crate::solution::Solution).

#[cfg(doc)]
use crate::objective::Objective;
#[cfg(doc)]
use crate::optimizer::Optimizer;
#[cfg(doc)]
use crate::solution::Solution;
#[cfg(doc)]
use crate::variable::var::Var;

use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

pub trait PreDomain{}

/// [`Domain`] is a trait describing the type of a point from the domain it is attached to.
/// It must implement the [`sample`](Domain::sample) and [`is_in`](Domain::is_in) methods.
///
/// # Notes
///
/// A [`Domain`] should always have a `::new(...)->Self` method.
/// This method is used in the [`objective!`](../../../tantale/macros/macro.objective.html) and [`sp!`](../../../tantale/macros/macro.sp.html) procedural macro.
pub trait Domain: PreDomain + Sized + PartialEq + Debug {
    /// [`TypeDom`](Domain::TypeDom) defines the type of a point sampled
    /// from the [`Domain`]. This is one of the main component defining
    /// most of the typing within the library.
    type TypeDom: Sized
        + PartialEq
        + Clone
        + Display
        + Debug
        + Default
        + Serialize
        + for<'a> Deserialize<'a>;
    /// Default sampling algorithm used to get a random point from
    /// the [`Domain`].
    ///
    /// # Parameters
    ///
    /// * `rng` : `&mut`[`ThreadRng`](rand::prelude::ThreadRng) - The RNG from [`rand`].
    ///
    fn sample(&self, rng: &mut ThreadRng) -> Self::TypeDom;
    /// Returns `true` if a given borrowed `point` is in the domain. Otherwise returns `false`.
    ///
    /// # Parameters
    ///
    /// * `point` : `&`[`Self`]`::`[`TypeDom`](Domain::TypeDom) - a borrowed point from the [`Domain`].
    ///
    fn is_in(&self, point: &Self::TypeDom) -> bool;
}

/// [`Mixed`] trait defines a [`Domain`] which can be made of other [`Domains`](Domain).
/// For example an `enum` of [`Domains`](Domain).
/// This trait is mainly used by the derive macro [`#[derive(Mixed)]`](../../../tantale/derive.Mixed.html).
pub trait Mixed: Domain {}

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

pub mod base;
pub use base::{BaseDom, BaseTypeDom};

pub mod onto;
pub use onto::Onto;