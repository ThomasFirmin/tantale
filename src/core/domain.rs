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

#[macro_export]
macro_rules! domain_obj {
    ($(let $var:ident $(: $type:ty)? = $name:ident::new($($args:expr),*)),+$(,)?) => {
        use $crate::core::domain::*;
        $(let $var $(: $type)? = $name::new($($args),*));+;
    };
}

#[macro_export]
macro_rules! domain_opt {
    ($(let $var:ident $(: $type:ty)? = $name:ident::new($lower:expr,$upper:expr)),+) => {
        use $crate::core::domain::*;
        $(let $var $(: $type)? = $name::new($lower,$upper));+;
    };
}

/// [`Domain`] is a trait describing the type of a point from the domain it is attached to.
/// It must implement the `default_sampler` and `is_in` methods.
pub trait Domain: Sized {
    type TypeDom: PartialEq + Clone + Copy + Display + Debug;
    /// Associated function to automatically return a default [`crate::core::sampler`]
    /// for the domain the trait is implemented.
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom;
    /// Returns `true` if a given borrowed `point` is in the domain. Otherwise returns `false`.
    ///
    /// # Parameters
    ///
    /// * `point` : `&`[`Self`]`::`[`TypeDom`](Domain::TypeDom) - a borrowed point from the [`Domain`].
    ///
    fn is_in(&self, point: &Self::TypeDom) -> bool;
}



pub mod bounded;
pub use bounded::{Bounded,DomainBounded,Real,Nat,Int};

pub mod unit;
pub use unit::Unit;

pub mod bool;
pub use bool::Bool;

pub mod cat;
pub use cat::Cat;

pub mod base;
pub use base::{BaseDom,BaseTypeDom};

pub mod onto;
pub use onto::Onto;

pub mod sampler;
pub use sampler::{uniform,uniform_real,uniform_nat,uniform_int,uniform_bool,uniform_cat};

pub mod errors_domain;
pub use errors_domain::{DomainError, DomainBoundariesError,DomainOoBError};