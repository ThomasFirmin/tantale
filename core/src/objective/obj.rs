//! The [`Objective`](tantale::core::Objective) describes the wrapper arround the function
//! the user wants to maximize. This function must output an [`Outcome`](tantale::core::Outcome)
//! which will be further processed by the [`Codomain`](tantale::core::Codomain).
//!

use crate::domain::{Domain, TypeDom};
use crate::objective::outcome::{FuncState, Outcome};
use crate::objective::Codomain;
use std::sync::Arc;

type OptimFn<TypeDom, Out> = fn(&[TypeDom]) -> Out;
type SteppFn<TypeDom, Out, FnState> = fn(&[TypeDom], Option<FnState>) -> (Out, FnState);

/// A wrapper arround the user-defined function to maximize.
pub trait FuncWrapper {}

/// Describes the fidelity state of a [`Partial`].
///
/// * [`New`](FidelState::New) : A newly created solution.
/// * [`Resume`](FidelState::Resume) : Resume the evaluation of [`Partial`].
/// * [`Discard`](FidelState::Discard) : Discard a [`Partial`] that has already been evaluated for a few steps.
#[derive(Copy, Clone)]
pub enum FidelState {
    New,
    Resume,
    Discard,
}

/// [`Objective`] is the minimal wrapper for the raw function to maximize.
/// This raw function must return an [`Outcome`],
/// according to an input `x` of type [`TypeDom`](tantale::core::Domain::TypeDom).
///
/// # Attributes
///
/// * `codomain` : `Cod` - A given [`Codomain`] extracted from an the function's [`Outcome`].
/// * `function` : `fn(&[Obj::TypeDom],Arc<Out>) -> Out` - A function to be maximized. It takes a vector
///    containing the point to be evaluated.
pub struct Objective<Obj, Cod, Out>
where
    Obj: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    pub codomain: Arc<Cod>,
    pub function: OptimFn<Obj::TypeDom, Out>,
}

impl<Obj, Cod, Out> Objective<Obj, Cod, Out>
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
    pub fn new(cod: Cod, func: OptimFn<Obj::TypeDom, Out>) -> Self {
        Self {
            codomain: Arc::new(cod),
            function: func,
        }
    }
    /// Initialize the ['Objective'].
    pub fn init(&mut self) {}
    /// Compute the raw outputs of a function to maximize according to an input `x`.
    pub fn raw_compute(&self, x: &[TypeDom<Obj>]) -> Out {
        (self.function)(x)
    }
    /// Compute the outputs of a function to maximize according to an input `x`.    
    pub fn compute(&self, x: &[TypeDom<Obj>]) -> (Arc<Cod::TypeCodom>, Arc<Out>) {
        let out = self.raw_compute(x);
        (Arc::new(self.codomain.get_elem(&out)), Arc::new(out))
    }

    pub fn get_codomain(&self) -> Arc<Cod> {
        self.codomain.clone()
    }
}

impl<Obj, Cod, Out> FuncWrapper for Objective<Obj, Cod, Out>
where
    Obj: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
}

/// The [`Stepped`] allows to define the minimal behavior of the wrapper.
/// The [`Objective`] must return a [`Codomain`]'s [`TypeCodom`](Codomain::TypeCodom), and an [`Outcome`],
/// according to an input `x` of type [`TypeDom`](tantale::core::Domain::TypeDom).
///
/// # Attributes
///
/// * `codomain` : `Cod` - A given [`Codomain`] extracted from an the function's [`Outcome`].
/// * `function` : `fn(&[Obj::TypeDom],Arc<Out>) -> Out` - A function to be maximized. it takes a vector containing the point to be evaluated, and an optional [`Outcome`]
///   previsouly computed in case of multi-fidelity optimization where function are evaluated by steps.
pub struct Stepped<Obj, Cod, Out, FnState>
where
    Obj: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    FnState: FuncState,
{
    pub codomain: Arc<Cod>,
    pub function: SteppFn<Obj::TypeDom, Out, FnState>,
}

impl<Obj, Cod, Out, FnState> Stepped<Obj, Cod, Out, FnState>
where
    Obj: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    FnState: FuncState,
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
    pub fn new(cod: Cod, func: SteppFn<Obj::TypeDom, Out, FnState>) -> Self {
        Self {
            codomain: Arc::new(cod),
            function: func,
        }
    }
    /// Initialize the ['Objective'].
    pub fn init(&mut self) {}
    /// Compute the raw outputs of a function to maximize according to an input `x`.
    pub fn raw_compute(&self, x: &[TypeDom<Obj>], state: Option<FnState>) -> (Out, FnState) {
        (self.function)(x, state)
    }
    /// Compute the outputs of a function to maximize according to an input `x`.    
    pub fn compute(
        &self,
        x: &[TypeDom<Obj>],
        state: Option<FnState>,
    ) -> (Arc<Cod::TypeCodom>, Arc<Out>, FnState) {
        let (out, state) = self.raw_compute(x, state);
        (Arc::new(self.codomain.get_elem(&out)), Arc::new(out), state)
    }

    pub fn get_codomain(&self) -> Arc<Cod> {
        self.codomain.clone()
    }
}

impl<Obj, Cod, Out, FnState> FuncWrapper for Stepped<Obj, Cod, Out, FnState>
where
    Obj: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    FnState: FuncState,
{
}
