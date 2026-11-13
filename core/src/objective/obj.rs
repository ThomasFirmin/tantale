//! Objective wrappers for user-defined functions.
//!
//! An objective is a thin wrapper around the function the user wants to optimize.
//! The raw function must return an [`Outcome`], from which
//! a [`Codomain`](crate::domain::codomain::Codomain) is extracted.
//!
//! # Examples
//! ## Single-shot objective
//! Wrap the following code inside a module, to use effortlessly with Tantale.
//! The `objective!` macro creates various functions
//! generating the searchspace, and the objective itself.
//! See the crate-level documentation for complete, tested usage examples.
//!
//! ## Multi-steps objective
//! Wrap the following code inside a module, to use effortlessly with Tantale.
//! The `objective!` macro creates various functions
//! generating the searchspace, and the objective itself.
//! Pay attention to the returned [`FuncState`], and [`EvalStep`](crate::objective::EvalStep).
//! See the crate-level documentation for complete, tested usage examples.
use serde::{Deserialize, Serialize};

use crate::{
    BaseSol, Domain, FidOutcome, FidelitySol, HasX, SolId, SolInfo, Solution, StepId, Uncomputed,
    objective::outcome::{FuncState, Outcome},
    solution::partial::Fidelity,
};
use std::{fmt::Debug, sync::Arc};

type OptimFn<Raw, Out> = fn(Raw) -> Out;
type SteppFn<Raw, Out, FnState> = fn(Raw, Fidelity, Option<FnState>) -> (Out, FnState);

/// Marker trait for wrappers around user-defined objective functions.
///
/// Both [`OneStep`] (single-shot evaluation) and [`MultiStep`] (multi-[`Step`](crate::objective::Step))
/// implement this trait so they can be used interchangeably by higher-level
/// components.
///
/// # Associated Derive Macro
///
/// The `FuncWrapper` derive macro automatically implements the trait for any struct
/// satisfying the required trait bounds.
pub trait FuncWrapper: Send + Sync + Clone {
    type Context;
    type Out: Outcome;
    type OutShape;
    type Dom: Domain;
    type Id: SolId;
    type Info: SolInfo;
    type MetaData: Clone + Default + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;
    type Sol<Dom: Domain>: Uncomputed<Dom = Dom, Id = Self::Id, Info = Self::Info>;
    /// Evaluate the objective at a given raw input `x` with a given `context`.
    fn compute(
        &self,
        x: <Self::Sol<Self::Dom> as HasX>::X,
        context: Self::Context,
    ) -> Self::OutShape;
}

/// Minimal wrapper for a single-shot objective function.
///
/// The wrapped function maps a raw solution input `x` to an [`Outcome`]. The
/// input is typically the searchspace solution representation.
///
/// # Type parameters
/// - `S`: [`Solution`] type, which provides the input type through [`HasX::X`].
/// - `Out`: Output implementing [`Outcome`].
pub struct OneStep<S: Solution, Out: Outcome>(pub OptimFn<S::X, Out>);

impl<S: Solution, Out: Outcome> OneStep<S, Out> {
    /// Creates a new [`OneStep`] wrapper from a raw function.
    ///
    /// # Parameters
    /// - `func`: The objective function to optimize.
    ///
    /// # See also
    /// - [`Searchspace`](crate::searchspace::Searchspace)
    /// - [`objective!`](../../../doc/tantale_macros/macro.objective.html)
    pub fn new(func: OptimFn<S::X, Out>) -> Self {
        OneStep(func)
    }
}

impl<S: Solution, Out: Outcome> Clone for OneStep<S, Out> {
    fn clone(&self) -> Self {
        OneStep(self.0)
    }
}

impl<Id, Obj, Info, Out> FuncWrapper for OneStep<BaseSol<Id, Obj, Info>, Out>
where
    Id: SolId,
    Obj: Domain,
    Info: SolInfo,
    Out: Outcome,
{
    type Context = ();
    type OutShape = Out;
    type Out = Out;
    type Dom = Obj;
    type Id = Id;
    type Info = Info;
    type MetaData = <BaseSol<Id, Obj, Info> as Solution>::MetaData;
    type Sol<Dom: Domain> = BaseSol<Id, Dom, Info>;

    fn compute(&self, x: Arc<[Obj::Element]>, _context: Self::Context) -> Self::OutShape {
        (self.0)(x)
    }
}

/// Wrapper for objectives evaluated by [`Step`](crate::objective::Step).
///
/// The wrapped function receives an input `x`, a [`Fidelity`], and an optional
/// [`FuncState`] from previous [`Step`](crate::objective::Step)s. It returns the [`Outcome`] and the updated
/// [`FuncState`].
///
/// # Type parameters
/// - `Raw`: Input type passed to the objective function.
/// - `Out`: Output implementing [`FidOutcome`].
/// - `FnState`: Internal function state implementing [`FuncState`].
pub struct MultiStep<S, Out, FnState>(pub SteppFn<S::X, Out, FnState>)
where
    S: Solution,
    Out: FidOutcome,
    FnState: FuncState;

impl<S, Out, FnState> Clone for MultiStep<S, Out, FnState>
where
    S: Solution,
    Out: FidOutcome,
    FnState: FuncState,
{
    fn clone(&self) -> Self {
        MultiStep(self.0)
    }
}

impl<S, Out, FnState> MultiStep<S, Out, FnState>
where
    S: Solution,
    Out: FidOutcome,
    FnState: FuncState,
{
    /// Creates a new [`MultiStep`] from a raw multi-[`Step`](crate::objective::Step) function.
    ///
    /// # Parameters
    /// - `func`: The stepped objective function to optimize.
    ///
    /// # See also
    /// - [`Searchspace`](crate::searchspace::Searchspace)
    /// - [`objective!`](../../../doc/tantale_macros/macro.objective.html) procedural macro
    pub fn new(func: SteppFn<S::X, Out, FnState>) -> Self {
        MultiStep(func)
    }
}

impl<Id, Obj, Info, Out, FnState> FuncWrapper
    for MultiStep<FidelitySol<Id, Obj, Info>, Out, FnState>
where
    Id: StepId,
    Obj: Domain,
    Info: SolInfo,
    Out: FidOutcome,
    FnState: FuncState,
{
    type Context = (Fidelity, Option<FnState>);
    type OutShape = (Out, FnState);
    type Out = Out;
    type Dom = Obj;
    type Id = Id;
    type Info = Info;
    type MetaData = <FidelitySol<Id, Obj, Info> as Solution>::MetaData;
    type Sol<Dom: Domain> = FidelitySol<Id, Dom, Info>;

    fn compute(&self, x: Arc<[Obj::Element]>, context: Self::Context) -> Self::OutShape {
        (self.0)(x, context.0, context.1)
    }
}
