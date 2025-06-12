//! The [`Objective`](tantale::core::Objective) describes the wrapper arround the function
//! the user wants to maximize. This function must output an [`Outcome`](tantale::core::Outcome)
//! which will be further processed by the [`Codomain`](tantale::core::Codomain).
//! The [`Codomain`](tantale::core::Codomain)
//!

use crate::domain::Domain;
use crate::objective::outcome::Outcome;
use crate::objective::Codomain;

use std::fmt::{Debug, Display};

/// The trait [`Objective`] allows to define the minimal behavior of the wrapper.
/// The [`Objective`] must return a [`Codomain`]'s [`TypeCodom`](Codomain::TypeCodom), and an [`Outcome`],
/// according to an input `x` of type [`TypeDom`](tantale::core::Domain::TypeDom).
///
pub trait Objective<Obj, Cod, Out>
where
    Obj: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    /// Compute the outputs of a function to maximize according to an input `x`.
    fn compute(&self, x: &[Obj::TypeDom]) -> (Cod::TypeCodom, Out);
}

/// A simple structure wrapping a user defined function to be maximized.
///
/// # Attributes
///
/// * `codomain` : `Cod` - A given [`Codomain`] extracted from an the function's [`Outcome`].
/// * `function` : `fn(&[Obj::TypeDom]) -> Out` - A function to be maximized.
pub struct SimpleObjective<Obj, Cod, Out>
where
    Obj: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    pub codomain: Cod,
    pub function: fn(&[Obj::TypeDom]) -> Out,
}

impl<Obj, Cod, Out> Objective<Obj, Cod, Out> for SimpleObjective<Obj, Cod, Out>
where
    Obj: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    fn compute(&self, x: &[Obj::TypeDom]) -> (Cod::TypeCodom, Out) {
        let out = (self.function)(x);
        (self.codomain.get_elem(&out), out)
    }
}
