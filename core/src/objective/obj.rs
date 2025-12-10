//! The [`Objective`](tantale::core::Objective) describes the wrapper arround the function
//! the user wants to maximize. This function must output an [`Outcome`](tantale::core::Outcome)
//! which will be further processed by the [`Codomain`](tantale::core::Codomain).
//!

use serde::{Deserialize, Serialize};

use crate::{
    FidOutcome, objective::outcome::{FuncState, Outcome}, solution::partial::Fidelity
};

type OptimFn<Raw, Out> = fn(Raw) -> Out;
type SteppFn<Raw, Out, FnState> = fn(Raw, Fidelity, Option<FnState>) -> (Out, FnState);

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
pub struct Objective<Raw, Out>(pub OptimFn<Raw, Out>)
where
    Raw: Serialize + for<'a> Deserialize<'a>,
    Out:Outcome;

impl<Raw, Out> Objective<Raw, Out>
where
    Raw: Serialize + for<'a> Deserialize<'a>,
    Out:Outcome,
{
    /// Creates an new instance of [`ObjBase`].
    ///
    /// # Parameters
    ///
    /// * `func` : The objective function to be optimized and defined by the user.
    ///   It can be created side-by-side with the [`Searchspace`] using the
    ///   [`hpo!`](tantale::macros:objective) macro.
    ///
    pub fn new(func: OptimFn<Raw, Out>) -> Self {
        Objective(func)
    }
    /// Initialize the ['Objective'].
    pub fn init(&mut self) {}
    /// Compute the raw outputs of a function to maximize according to an input `x`.
    pub fn compute(&self, x: Raw) -> Out {
        (self.0)(x)
    }
}

impl<Raw, Out> FuncWrapper for Objective<Raw, Out> 
where
    Raw: Serialize + for<'a> Deserialize<'a>,
    Out:Outcome
{}

/// The [`Stepped`] allows to define the minimal behavior of the wrapper.
/// The [`Objective`] must return an [`Outcome`],
/// according to an input `x` of type [`TypeDom`](tantale::core::Domain::TypeDom).
///
/// # Attributes
///
/// * `function` : `fn(&[Obj::TypeDom],Arc<Out>) -> Out` - A function to be maximized. it takes a vector containing the point to be evaluated, and an optional [`Outcome`]
///   previsouly computed in case of multi-fidelity optimization where function are evaluated by steps.
pub struct Stepped<Raw, Out, FnState>(pub SteppFn<Raw, Out, FnState>)
where
    Raw: Serialize + for<'a> Deserialize<'a>,
    Out:FidOutcome,
    FnState:FuncState;

impl<Raw, Out, FnState> Stepped<Raw, Out, FnState>
where
    Raw: Serialize + for<'a> Deserialize<'a>,
    Out:FidOutcome,
    FnState:FuncState
{
    /// Creates an new instance of [`ObjBase`].
    ///
    /// # Parameters
    ///
    /// * `func` : The objective function to be optimized and defined by the user.
    ///   It can be created side-by-side with the [`Searchspace`] using the
    ///   [`hpo!`](tantale::macros:objective) macro.
    ///
    pub fn new(func: SteppFn<Raw, Out, FnState>) -> Self {
        Stepped(func)
    }
    /// Initialize the ['Objective'].
    pub fn init(&mut self) {}
    /// Compute the raw outputs of a function to maximize according to an input `x`.
    pub fn compute(
        &self,
        x: Raw,
        fidelity: Fidelity,
        state: Option<FnState>,
    ) -> (Out, FnState) {
        (self.0)(x, fidelity, state)
    }
}

impl<Raw, Out, FnState> FuncWrapper for Stepped<Raw, Out, FnState>
where
    Raw: Serialize + for<'a> Deserialize<'a>,
    Out:FidOutcome,
    FnState:FuncState
{}
