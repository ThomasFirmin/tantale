//! The [`Objective`](tantale::core::Objective) describes the wrapper arround the function
//! the user wants to maximize. This function must output an [`Outcome`](tantale::core::Outcome)
//! which will be further processed by the [`Codomain`](tantale::core::Codomain).
//! The [`Codomain`](tantale::core::Codomain)
//!

use crate::domain::{Domain, TypeDom};
use crate::objective::outcome::Outcome;
use crate::objective::Codomain;

use std::sync::Arc;

/// The trait [`Objective`] allows to define the minimal behavior of the wrapper.
/// The [`Objective`] must return a [`Codomain`]'s [`TypeCodom`](Codomain::TypeCodom), and an [`Outcome`],
/// according to an input `x` of type [`TypeDom`](tantale::core::Domain::TypeDom).
///
pub trait Objective<Obj, Cod, Out>
where
    Obj: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    /// Initialize the ['Objective'].
    fn init(&mut self);
    /// Compute the outputs of a function to maximize according to an input `x`.
    fn compute(&self, x: Arc<[TypeDom<Obj>]>) -> (Arc<Cod::TypeCodom>, Arc<Out>);

    fn get_codomain(&self) -> Arc<Cod>;
}

/// A simple structure wrapping a user defined function to be maximized.
///
/// # Attributes
///
/// * `codomain` : `Cod` - A given [`Codomain`] extracted from an the function's [`Outcome`].
/// * `function` : `fn(&[Obj::TypeDom]) -> Out` - A function to be maximized.
pub struct ObjBase<Obj, Cod, Out>
where
    Obj: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    pub codomain: Arc<Cod>,
    pub function: fn(Arc<[TypeDom<Obj>]>) -> Out,
}

impl<Obj, Cod, Out> ObjBase<Obj, Cod, Out>
where
    Obj: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    /// Creates an new instance of [`ObjBase`].
    ///
    /// # Parameters
    ///
    /// * `cod`  :  `Cod` -  A [`Codomain`] of a corresponding [`Outcome`].
    /// * `func` : The objective function to be optimized and defined by the user.
    ///   It can be created side-by-side with the [`Searchspace`] using the
    ///   [`objective!`](tantale::macros:objective) macro.
    ///
    pub fn new(cod: Cod, func: fn(Arc<[TypeDom<Obj>]>) -> Out) -> Self {
        Self {
            codomain: Arc::new(cod),
            function: func,
        }
    }
}
impl<Obj, Cod, Out> Objective<Obj, Cod, Out> for ObjBase<Obj, Cod, Out>
where
    Obj: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    fn init(&mut self) {}
    fn compute(&self, x: Arc<[TypeDom<Obj>]>) -> (Arc<Cod::TypeCodom>, Arc<Out>) {
        let out = (self.function)(x);
        (Arc::new(self.codomain.get_elem(&out)), Arc::new(out))
    }

    fn get_codomain(&self) -> Arc<Cod> {
        self.codomain.clone()
    }
}
