//! `Has`-like traits provide a unified interface for retrieving information such as solution identifiers, raw solution data, objective function
//! values, evaluation steps, fidelity levels, and associated metadata, regardless of the specific solution representation used in the optimization process.

use serde::{Deserialize, Serialize};

use crate::{
    EvalStep, Fidelity, Linked, SolId, SolInfo, State, Step, StepId, Uncomputed, Var,
};
use std::{fmt::Debug, sync::Arc};

/// Trait for objects with a unique solution identifier.
///
/// [`HasId`] provides access to a solution's unique [`Id`], which remains constant across
/// domain transformations. Twin solutions (same point in different domains) share the same [`Id`],
/// enabling tracking and correlation throughout the optimization process.
pub trait HasId 
{
    type Id: SolId;
    /// Returns the solution's unique identifier.
    fn id(&self) -> Self::Id;

    /// Returns a reference to the solution's unique identifier.
    fn ref_id(&self) -> &Self::Id;

    /// Returns a mutable reference to the solution's unique identifier.
    fn mut_ref_id(&mut self) -> &mut Self::Id;

    /// Checks if another object is a twin (shares the same [`Id`]).
    ///
    /// Two solutions are twins if they represent the same point in different domains
    /// or shapes. Twin solutions always have equal [`Id`]s.
    ///
    /// # Parameters
    ///
    /// * `solb` - Another object with an [`Id`] to compare
    ///
    /// # Returns
    ///
    /// `true` if both objects share the same [`Id`], `false` otherwise.
    fn is_twin<Twin: HasId<Id = Self::Id>>(&self, solb: Twin) -> bool {
        self.ref_id() == solb.ref_id()
    }
}

/// Trait for objects with a unique solution identifier.
///
/// [`HasStepId`] extends [`HasId`] to provide access to a solution's unique identifier that also tracks
/// how many times that solution has been passed to the function to optimize.
pub trait HasStepId: HasId 
where
    Self::Id: StepId
{
    /// Increments the step counter in the solution's identifier.
    fn increment(&mut self);

    /// Returns the current step count from the solution's identifier.
    fn id_step(&self) -> usize;

    /// Returns a new identifier representing the previous step id.
    fn previous_id(&self) -> Self::Id;
}

/// Trait for objects carrying solution metadata.
///
/// [`HasInfo`] provides access to a solution's associated metadata ([`SolInfo`]), which may
/// include iteration numbers, timestamps, or other optimizer-specific information specifically related to the solution.
pub trait HasInfo
{
    type Info: SolInfo;
    /// Returns the solution's [`SolInfo`] wrapped in [`Arc`].
    fn sinfo(&self) -> Arc<Self::Info>;
}

/// Trait for objects containing a raw solution.
///
/// [`HasX`] provides access to the raw solution data, which is the internal representation of the solution.
/// This trait is typically implemented by both [`Uncomputed`] and [`Computed`](crate::Computed) solutions,
/// allowing retrieval of the underlying raw solution regardless of its computed state.
pub trait HasX 
{
    type X: Send + Sync + Clone + Debug + Serialize + for<'de> Deserialize<'de>;
    /// Returns a reference to the raw solution associated with this object.
    fn ref_x(&self) -> &Self::X;

    /// Returns a clone of the raw solution associated with this object.
    fn clone_x(&self) -> Self::X;
}

/// Trait for objects with an associated objective function value.
///
/// [`HasY`] provides access to a solution's evaluation result ([`TypeCodom`](crate::Codomain::TypeCodom)), representing the output of the
/// objective function. This trait is typically implemented by [`Computed`](crate::Computed) solutions.
///
/// # Note
///
/// When the [`TypeCodom`](crate::Codomain::TypeCodom) is [`Ord`], [`PartialOrd`], [`Eq`] or [`PartialEq`]
/// (e.g. [`SingleCodomain`](crate::SingleCodomain)), then objects implementing [`HasY`], such as
/// [`Computed`](crate::Computed), [`Pair`](crate::Pair), [`Lone`](crate::Lone), are also [`Ord`], [`PartialOrd`], [`Eq`] or [`PartialEq`] respectively.
pub trait HasY
{
    type Y: Debug + Serialize + for<'a> Deserialize<'a>;

    /// Returns the objective function value associated with this solution.
    ///
    /// # Returns
    ///
    /// A shared reference to the codomain value (objective function output).
    fn y(&self) -> Arc<Self::Y>;
}

/// Trait for solutions supporting multi-fidelity evaluation tracking.
///
/// [`HasStep`] enables tracking evaluation progress for objective functions that can be
/// evaluated incrementally (multi-fidelity). The evaluation state is represented by a [`Step`],
/// which can indicate pending evaluation, partial progress, completion, or discard status.
///
/// # Evaluation States
///
/// - **[`Pending`](Step::Pending)**: Awaiting evaluation
/// - **[`Partially(n)`](Step::Partially)**: Evaluated to step `n`
/// - **[`Evaluated`](Step::Evaluated)**: Fully evaluated
/// - **[`Discard`](Step::Discard)**: Marked for rejection without full evaluation
/// - **[`Error`](Step::Error)**: Evaluation failed
pub trait HasStep {
    /// Returns the current evaluation [`Step`].
    fn step(&self) -> Step;

    /// Returns the raw internal evaluation [`EvalStep`].
    fn raw_step(&self) -> EvalStep;

    /// Sets the evaluation state directly via a raw [`EvalStep`].
    fn set_raw_step(&mut self, value: EvalStep);

    /// Sets the evaluation state via a high-level [`Step`].
    fn set_step(&mut self, value: Step) {
        self.set_raw_step(value.into());
    }

    /// Marks the solution as pending evaluation.
    ///
    /// Sets the state to [`Pending`](Step::Pending), indicating the solution
    /// has not yet been evaluated.
    fn pending(&mut self);

    /// Marks the solution as partially evaluated to a specific step.
    ///
    /// # Parameters
    ///
    /// * `value` - The step number reached (non-negative)
    fn partially(&mut self, value: isize);

    /// Marks the solution for discard.
    ///
    /// Sets the state to [`Discard`](Step::Discard), indicating the solution
    /// should be rejected without further evaluation (e.g., early stopping).
    fn discard(&mut self);

    /// Marks the solution as fully evaluated.
    ///
    /// Sets the state to [`Evaluated`](Step::Evaluated), indicating the solution
    /// has been completely evaluated at the highest fidelity.
    fn evaluated(&mut self);

    /// Marks the solution as having encountered an error during evaluation.
    ///
    /// Sets the state to [`Error`](Step::Error), indicating evaluation failed.
    fn error(&mut self);
}

/// Trait for solutions with an associated fidelity level.
///
/// [`HasFidelity`] tracks the computational budget or resource level at which a solution
/// was or should be evaluated. This is used in multi-fidelity optimization to control
/// evaluation costs and implement progressive evaluation strategies.
///
/// Higher fidelity values generally mean more accurate but more expensive evaluations.
pub trait HasFidelity {
    /// Returns the current fidelity level.
    fn fidelity(&self) -> Fidelity;

    /// Sets the fidelity level for this solution.
    fn set_fidelity(&mut self, fidelity: f64);
}

/// Trait for objects containing an uncomputed solution.
///
/// [`HasUncomputed`] provides access to the underlying [`Uncomputed`] solution within
/// wrapper types like [`Computed`](crate::Computed) or [`SolutionShape`](crate::SolutionShape).
pub trait HasUncomputed
{
    /// The uncomputed solution type contained within.
    type Uncomputed: Uncomputed;

    /// Returns a reference to the contained uncomputed solution.
    fn get_uncomputed(&self) -> &Self::Uncomputed;
}

/// Trait for objects containing a slice of [Var]iables.
pub trait HasVariables: Linked {

    fn get_vars(&self) -> &[Var<Self::Obj, Self::TrueOpt>];

    /// Returns an iterator over the `Opt` [`Domain`]s associated with this object.
    fn iter_opt(&self) -> impl Iterator<Item = &Self::Opt> + '_ {
        (0..self.size()).map(|idx| self.opt_at(idx).unwrap())
    }

    /// Returns an iterator over the `Obj` [`Domain`]s associated with this object.
    fn iter_obj(&self) -> impl Iterator<Item = &Self::Obj> + '_ {
        (0..self.size()).map(|idx| self.obj_at(idx).unwrap())
    }

    /// Returns the [`Var`] at a specific index, if it exists.
    fn var_at(&self, index: usize) -> Option<&Var<Self::Obj, Self::TrueOpt>> {
        self.get_vars().get(index)
    }

    /// Returns the `Obj` [`Domain`] at a specific index, if it exists.
    fn obj_at(&self, index: usize) -> Option<&Self::Obj>;

    /// Returns the `Opt` [`Domain`] at a specific index, if it exists.
    fn opt_at(&self, index: usize) -> Option<&Self::Opt>;

    /// Returns the number of variables associated with this object.
    fn size(&self) -> usize;

    /// Returns the names of the variables associated with this object.
    fn names(&self) -> Vec<String>;
}


/// Trait for objects that can provide their internal state.
pub trait HasState 
{
    type State: State;

    /// Returns a reference to the [`State`] of `Self`.
    fn get_state(&self) -> &Self::State;
    /// Returns a mutable reference to the [`State`] of `Self`.
    fn get_mut_state(&mut self) -> &mut Self::State;
    /// Builds `Self` from a previously saved [`State`].
    fn from_state(state: Self::State) -> Self;
}

/// Trait for objects that can provide their batch size.
pub trait HasBatchSize {
    /// Sets the batch size for the optimizer.
    fn set_batch_size(&mut self, batch_size: usize);

    /// Returns the current batch size of the optimizer.
    fn get_batch_size(&self) -> usize;
}