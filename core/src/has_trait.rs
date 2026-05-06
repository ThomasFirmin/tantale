use crate::{Codomain, Domain, EvalStep, Fidelity, Id, Linked, OptInfo, Outcome, SolInfo, Step, StepId, Uncomputed, Var};
use std::sync::Arc;

/// Trait for objects with a unique solution identifier.
///
/// [`HasId`] provides access to a solution's unique [`Id`], which remains constant across
/// domain transformations. Twin solutions (same point in different domains) share the same [`Id`],
/// enabling tracking and correlation throughout the optimization process.
pub trait HasId<SolId: Id> {
    /// Returns the solution's unique identifier.
    fn id(&self) -> SolId;

    /// Returns a reference to the solution's unique identifier.
    fn ref_id(&self) -> &SolId;

    /// Returns a mutable reference to the solution's unique identifier.
    fn mut_ref_id(&mut self) -> &mut SolId;

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
    fn is_twin<Twin: HasId<SolId>>(&self, solb: Twin) -> bool {
        self.ref_id() == solb.ref_id()
    }
}


/// Trait for objects with a unique solution identifier.
///
/// [`HasStepId`] extends [`HasId`] to provide access to a solution's unique identifier that also tracks
/// how many times that solution has been passed to the function to optimize.
pub trait HasStepId<SolId: StepId>: HasId<SolId> {
    /// Increments the step counter in the solution's identifier.
    fn increment(&mut self);

    /// Returns the current step count from the solution's identifier.
    fn id_step(&self) -> usize;

    /// Returns a new identifier representing the previous step id.
    fn previous_id(&self) -> SolId;
}

/// Trait for objects carrying solution metadata.
///
/// [`HasSolInfo`] provides access to a solution's associated metadata ([`SolInfo`]), which may
/// include iteration numbers, timestamps, or other optimizer-specific information specifically related to the solution.
pub trait HasSolInfo<Info: SolInfo> {
    /// Returns the solution's [`SolInfo`] wrapped in [`Arc`].
    fn sinfo(&self) -> Arc<Info>;
}

/// Trait for objects with an associated objective function value.
///
/// [`HasY`] provides access to a solution's evaluation result ([`TypeCodom`](Codomain::TypeCodom)), representing the output of the
/// objective function. This trait is typically implemented by [`Computed`] solutions.
///
/// # Note
///
/// When the [`TypeCodom`](Codomain::TypeCodom) is [`Ord`], [`PartialOrd`], [`Eq`] or [`PartialEq`]
/// (e.g. [`SingleCodomain`](crate::SingleCodomain)), then objects implementing [`HasY`], such as
/// [`Computed`], [`Pair`], [`Lone`], are also [`Ord`], [`PartialOrd`], [`Eq`] or [`PartialEq`] respectively.
pub trait HasY<Cod: Codomain<Out>, Out: Outcome> {
    /// Returns the objective function value associated with this solution.
    ///
    /// # Returns
    ///
    /// A shared reference to the codomain value (objective function output).
    fn y(&self) -> Arc<Cod::TypeCodom>;
}

/// Trait for objects carrying optimizer-specific metadata.
///
/// [`HasInfo`] provides access to optimizer-specific information that may be attached to
/// solutions, distinct from general [`SolInfo`].
pub trait HasInfo<Info: OptInfo> {
    /// Returns the optimizer-specific metadata for this solution.
    fn info(&self) -> Arc<Info>;
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
/// wrapper types like [`Computed`] or [`SolutionShape`].
pub trait HasUncomputed<SolId: Id, Dom: Domain, SInfo: SolInfo> {
    /// The uncomputed solution type contained within.
    type Uncomputed: Uncomputed<SolId, Dom, SInfo>;

    /// Returns a reference to the contained uncomputed solution.
    fn get_uncomputed(&self) -> &Self::Uncomputed;
}

/// Trait for objects containing a slice of [`Var`]iables.
pub trait HasVariables: Linked
{
    /// Returns a vector of variables associated with this object.
    fn variables(&self) -> &[Var<Self::Obj, Self::TrueOpt>];

    /// Returns the `Obj` [`Domain`] at a specific index, if it exists.
    fn obj_at(&self, index: usize) -> Option<&Self::Obj>;

    /// Returns the `Opt` [`Domain`] at a specific index, if it exists.
    fn opt_at(&self, index: usize) -> Option<&Self::Opt>;

    /// Returns the number of variables associated with this object.
    fn size(&self) -> usize {
        self.variables().len()
    }

}