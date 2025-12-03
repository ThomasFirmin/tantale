//! The [`Objective`](tantale::core::Objective) describes the wrapper arround the function
//! the user wants to maximize. This function must output an [`Outcome`](tantale::core::Outcome)
//! which will be further processed by the [`Codomain`](tantale::core::Codomain).
//!

use crate::{
    FidOutcome, domain::{Domain, TypeDom}, objective::outcome::{FuncState, Outcome}, solution::partial::Fidelity
};

type OptimFn<Obj, Out> = fn(&[TypeDom<Obj>]) -> Out;
type SteppFn<Obj, Out, FnState> = fn(&[TypeDom<Obj>], Fidelity, Option<FnState>) -> (Out, FnState);

/// A wrapper arround the user-defined function to maximize.
pub trait FuncWrapper {}

/// [`Objective`] is the minimal wrapper for the raw function to maximize.
/// This raw function must return an [`Outcome`],
/// according to an input `x` of type [`TypeDom`](tantale::core::Domain::TypeDom).
///
/// # Attributes
///
/// * `function` : `fn(&[Obj::TypeDom],Arc<Out>) -> Out` - A function to be maximized. It takes a vector
///   containing the point to be evaluated.
pub struct Objective<Obj, Out>(pub OptimFn<Obj::TypeDom, Out>)
where
    Obj: Domain,
    Out: Outcome;

impl<Obj, Out> Objective<Obj, Out>
where
    Obj: Domain,
    Out: Outcome,
{
    /// Creates an new instance of [`ObjBase`].
    ///
    /// # Parameters
    ///
    /// * `func` : The objective function to be optimized and defined by the user.
    ///   It can be created side-by-side with the [`Searchspace`] using the
    ///   [`hpo!`](tantale::macros:objective) macro.
    ///
    pub fn new(func: OptimFn<Obj::TypeDom, Out>) -> Self {
        Objective(func)
    }
    /// Initialize the ['Objective'].
    pub fn init(&mut self) {}
    /// Compute the raw outputs of a function to maximize according to an input `x`.
    pub fn compute(&self, x: &[TypeDom<Obj>]) -> Out {
        (self.0)(x)
    }
}

impl<Obj, Out> FuncWrapper for Objective<Obj, Out>
where
    Obj: Domain,
    Out: Outcome,
{
}

/// The [`Stepped`] allows to define the minimal behavior of the wrapper.
/// The [`Objective`] must return an [`Outcome`],
/// according to an input `x` of type [`TypeDom`](tantale::core::Domain::TypeDom).
///
/// # Attributes
///
/// * `function` : `fn(&[Obj::TypeDom],Arc<Out>) -> Out` - A function to be maximized. it takes a vector containing the point to be evaluated, and an optional [`Outcome`]
///   previsouly computed in case of multi-fidelity optimization where function are evaluated by steps.
pub struct Stepped<Obj, Out, FnState>(pub SteppFn<Obj::TypeDom, Out, FnState>)
where
    Obj: Domain,
    Out: FidOutcome,
    FnState: FuncState;

impl<Obj, Out, FnState> Stepped<Obj, Out, FnState>
where
    Obj: Domain,
    Out: FidOutcome,
    FnState: FuncState,
{
    /// Creates an new instance of [`ObjBase`].
    ///
    /// # Parameters
    ///
    /// * `func` : The objective function to be optimized and defined by the user.
    ///   It can be created side-by-side with the [`Searchspace`] using the
    ///   [`hpo!`](tantale::macros:objective) macro.
    ///
    pub fn new(func: SteppFn<Obj::TypeDom, Out, FnState>) -> Self {
        Stepped(func)
    }
    /// Initialize the ['Objective'].
    pub fn init(&mut self) {}
    /// Compute the raw outputs of a function to maximize according to an input `x`.
    pub fn compute(
        &self,
        x: &[TypeDom<Obj>],
        fidelity: Fidelity,
        state: Option<FnState>,
    ) -> (Out, FnState) {
        (self.0)(x, fidelity, state)
    }
}

impl<Obj, Out, FnState> FuncWrapper for Stepped<Obj, Out, FnState>
where
    Obj: Domain,
    Out: FidOutcome,
    FnState: FuncState,
{
}
