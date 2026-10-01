//! Solution shapes linking twins.
//!
//! A [`SolutionShape`] links two solutions with potentially different underlying domains or
//! types. The two solutions are [`twin`](crate::solution::Solution::twin)s and share the same [`Id`], letting users
//! manipulate a single object without separating objective- and optimizer-side representations.
//!
//! # Examples
//! ```
//! use tantale::core::{BaseSol, EmptyInfo, HasId, Id, Pair, Real, SId, SolutionShape, Unit, Uncomputed};
//! use std::sync::Arc;
//!
//! let info = Arc::new(EmptyInfo {});
//! let obj = BaseSol::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]), info.clone());
//! let opt = BaseSol::<SId, Unit, _>::new(obj.id(), Arc::from(vec![0.8]), info);
//! let pair = Pair::new(obj, opt);
//!
//! assert_eq!(pair.get_sobj().id(), pair.id());
//! assert_eq!(pair.get_sopt().id(), pair.id());
//! assert_eq!(pair.get_sobj().id(), pair.get_sopt().id());
//! ```

use crate::{
    Computed, Dominate, EvalStep, Fidelity, HasFidelity, HasId, HasInfo, HasStep, HasStepId, HasY, NoDomain, Outcome, Solution, StepId, 
    aliases::TypeCodom, domain::{codomain::ElemMulti, onto::Linked}, 
    objective::Step, solution::Uncomputed, utils::orderable::Orderable,
};

use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, fmt::Debug, sync::Arc};


/// Trait linking objective- and optimizer-side solutions with a shared [`Id`].
pub trait SolutionShape:
    Debug
    + HasId
    + HasInfo
    + Linked
    + Serialize
    + for<'de> Deserialize<'de>
{
    type ObjSol : Solution<Id = Self::Id, Info = Self::Info>;
    type OptSol : Solution<Id = Self::Id, Info = Self::Info>;

    /// Get a reference to the objective-side solution.
    fn get_sobj(&self) -> &Self::ObjSol;
    /// Get a reference to the optimizer-side solution.
    fn get_sopt(&self) -> &Self::OptSol;
    /// Get a mutable reference to the objective-side solution.
    fn get_mut_sobj(&mut self) -> &mut Self::ObjSol;
    /// Get a mutable reference to the optimizer-side solution.
    fn get_mut_sopt(&mut self) -> &mut Self::OptSol;
    /// Extract the objective-side solution, consuming the shape.
    fn extract_sobj(self) -> Self::ObjSol;
    /// Extract the optimizer-side solution, consuming the shape.
    fn extract_sopt(self) -> Self::OptSol;

    /// Creates a clone of this shape with the same solutions, metadata, and [`Id`], used
    /// for [`Accumulator`](crate::domain::codomain::Accumulator).
    fn _clone_shape(&self) -> Self;
}

pub trait UncomputedShape: SolutionShape
where
    Self::ObjSol: Uncomputed,
    Self::OptSol: Uncomputed,
{
}

pub trait IntoComputedShape: UncomputedShape
where
    Self::ObjSol: Uncomputed,
    Self::OptSol: Uncomputed,
{
    /// The computed type wrapping `Self` with an objective value.
    ///
    /// This type must implement [`HasY`] to provide access to the objective function output.
    type ComputedShape<Out:Outcome>: SolutionShape<
        Obj = Self::Obj,
        Opt = Self::Opt,
        TrueOpt = Self::TrueOpt,
        ObjSol = Computed<Self::ObjSol, Out>,
        OptSol = Computed<Self::OptSol, Out>,
        Id = Self::Id,
        Info = Self::Info,
    > + HasY<Y = TypeCodom<Out>>;

    /// Converts this uncomputed solution into a computed one with an objective value.
    ///
    /// # Type Parameters
    ///
    /// * `Cod` - The [`crate::Codomain`] defining the output structure
    /// * `Out` - The outcome type from the objective function
    ///
    /// # Parameters
    ///
    /// * `y` - The objective function output to associate with this solution
    ///
    /// # Returns
    ///
    /// A computed solution containing both `self` and `y`.
    fn into_computed<Out:Outcome>(self, y: Arc<TypeCodom<Out>>) -> Self::ComputedShape<Out>;

    /// Extracts the uncomputed solution and objective value from a computed solution.
    ///
    /// This is the inverse operation of [`into_computed`](IntoComputedShape::into_computed),
    /// decomposing a computed solution back into its constituent parts.
    ///
    /// # Type Parameters
    ///
    /// * `Out` - The outcome type from the objective function
    ///
    /// # Parameters
    ///
    /// * `comp` - The computed solution to decompose
    ///
    /// # Returns
    ///
    /// A tuple of `(uncomputed_solution, objective_value)`.
    fn extract<Out:Outcome>(comp: Self::ComputedShape<Out>) -> (Self, Arc<TypeCodom<Out>>);
}

/// A pair made of an objective [`Solution`] and its optimizer twin.
///
/// # Example
/// ```
/// use tantale::core::{SolutionShape, Uncomputed, BaseSol, EmptyInfo, HasId, Pair, Real, Id, SId, Unit};
/// use std::sync::Arc;
///
/// let info = Arc::new(EmptyInfo {});
/// let obj = BaseSol::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]), info.clone());
/// let opt = BaseSol::<SId, Unit, _>::new(obj.id(), Arc::from(vec![0.9]), info);
/// let pair = Pair::new(obj, opt);
///
/// assert_eq!(pair.get_sobj().id(), pair.id());
/// assert_eq!(pair.get_sopt().id(), pair.id());
/// assert_eq!(pair.get_sobj().id(), pair.get_sopt().id());
/// ```
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "SA: Serialize, SB: Serialize",
    deserialize = "SA: for<'a> Deserialize<'a>, SB: for<'a> Deserialize<'a>",
))]
pub struct Pair<SA, SB>(SA, SB)
where
    SA: Solution,
    SB: Solution<Id = SA::Id, Info = SA::Info>;

impl<SA, SB> Pair<SA, SB>
where
    SA: Solution,
    SB: Solution<Id = SA::Id, Info = SA::Info>
{
    /// Create a new paired solution from an objective and optimizer solution.
    pub fn new(sol_a: SA, sol_b: SB) -> Self {
        Self(sol_a,sol_b)
    }
}

impl<SA, SB> Linked for Pair<SA, SB>
where
    SA: Solution,
    SB: Solution<Id = SA::Id, Info = SA::Info>
{
    type Obj = SA::Dom;
    type Opt = SB::Dom;
    type TrueOpt = SB::Dom;
}

impl<SA, SB> HasId for Pair<SA, SB>
where
    SA: Solution,
    SB: Solution<Id = SA::Id, Info = SA::Info>
{
    type Id = SA::Id;

    fn id(&self) -> SA::Id {
        self.0.id()
    }

    fn ref_id(&self) -> &SA::Id {
        self.0.ref_id()
    }

    fn mut_ref_id(&mut self) -> &mut SA::Id {
        self.0.mut_ref_id()
    }
}

impl<SA, SB> HasStepId for Pair<SA, SB>
where
    SA: Solution,
    SB: Solution<Id = SA::Id, Info = SA::Info>,
    SA::Id: StepId,
    SB::Id: StepId,
{
    fn increment(&mut self) {
        self.0.id().increment();
        self.1.id().increment();
    }

    fn id_step(&self) -> usize {
        self.0.id().id_step()
    }

    fn previous_id(&self) -> SA::Id {
        self.0.id().previous_id()
    }
}

impl<SA, SB> HasInfo for Pair<SA, SB>
where
    SA: Solution,
    SB: Solution<Id = SA::Id, Info = SA::Info>
{
    type Info = SA::Info;
    fn sinfo(&self) -> Arc<SA::Info> {
        self.0.sinfo()
    }
}

impl<SA, SB> HasY for Pair<SA, SB>
where
    SA: Solution + HasY,
    SB: Solution<Id = SA::Id, Info = SA::Info> + HasY<Y = SA::Y>,
{
    type Y = SA::Y;
    fn y(&self) -> Arc<SA::Y> {
        self.0.y()
    }
}

impl<SA, SB> HasStep for Pair<SA, SB>
where
    SA: Solution + HasStep,
    SB: Solution<Id = SA::Id, Info = SA::Info> + HasStep,
{
    fn step(&self) -> Step {
        self.0.step()
    }

    fn raw_step(&self) -> EvalStep {
        self.0.raw_step()
    }

    fn pending(&mut self) {
        self.0.pending();
        self.1.pending();
    }

    fn partially(&mut self, value: isize) {
        self.0.partially(value);
        self.1.partially(value);
    }

    fn discard(&mut self) {
        self.0.discard();
        self.1.discard();
    }

    fn evaluated(&mut self) {
        self.0.evaluated();
        self.1.evaluated();
    }

    fn error(&mut self) {
        self.0.error();
        self.1.error();
    }

    fn set_raw_step(&mut self, value: EvalStep) {
        self.0.set_raw_step(value);
        self.1.set_raw_step(value);
    }
}

impl<SA, SB> HasFidelity for Pair<SA, SB>
where
    SA: Solution + HasFidelity,
    SB: Solution<Id = SA::Id, Info = SA::Info> + HasFidelity,
{
    fn fidelity(&self) -> Fidelity {
        self.0.fidelity()
    }

    fn set_fidelity(&mut self, fidelity: f64) {
        self.0.set_fidelity(fidelity);
        self.1.set_fidelity(fidelity);
    }
}

impl<SA, SB> SolutionShape for Pair<SA, SB>
where
    SA: Solution,
    SB: Solution<Id = SA::Id, Info = SA::Info>,
{
    type ObjSol = SA;
    type OptSol = SB;

    fn get_sobj(&self) -> &SA {
        &self.0
    }
    fn get_sopt(&self) -> &SB {
        &self.1
    }
    fn get_mut_sobj(&mut self) -> &mut SA {
        &mut self.0
    }
    fn get_mut_sopt(&mut self) -> &mut SB {
        &mut self.1
    }
    fn extract_sobj(self) -> SA {
        self.0
    }
    fn extract_sopt(self) -> SB {
        self.1
    }
    fn _clone_shape(&self) -> Self {
        let obj = self.get_sobj()._clone_sol();
        let opt = self.get_sopt()._clone_sol();
        Self::new(obj, opt)
    }
}

impl<SA, SB> UncomputedShape for Pair<SA, SB>
where
    SA: Uncomputed,
    SB: Uncomputed<Id = SA::Id, Info = SA::Info>,
{
}

impl<SA, SB> IntoComputedShape for Pair<SA, SB>
where
    SA: Uncomputed,
    SB: Uncomputed<Id = SA::Id, Info = SA::Info>,
{
    type ComputedShape<Out:Outcome> = Pair<Computed<SA, Out>, Computed<SB, Out>>;

    fn into_computed<Out:Outcome>(self, y: Arc<TypeCodom<Out>>) -> Self::ComputedShape<Out> {
        let cobj = Computed::new(self.0, y.clone());
        let copt = Computed::new(self.1, y);
        Pair::new(cobj, copt)
    }

    fn extract<Out:Outcome>(comp: Self::ComputedShape<Out>) -> (Self, Arc<TypeCodom<Out>>) {
        let y = comp.0.y;
        let pair = Pair::new(comp.0.sol, comp.1.sol);
        (pair, y)
    }
}

impl<SA, SB> PartialEq for Pair<SA, SB>
where
    Self: HasY<Y = SA::Y>,
    SA: Solution + HasY,
    SB: Solution<Id = SA::Id, Info = SA::Info> + HasY<Y = SA::Y>,
    SA::Y: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.y() == other.y()
    }
}

impl<SA,SB> Eq for Pair<SA, SB>
where
    Self: HasY<Y = SA::Y>,
    SA: Solution + HasY,
    SB: Solution<Id = SA::Id, Info = SA::Info> + HasY<Y = SA::Y>,
    SA::Y: Eq,
{
}

impl<SA,SB> PartialOrd for Pair<SA, SB>
where
    Self: HasY<Y = SA::Y>,
    SA: Solution + HasY,
    SB: Solution<Id = SA::Id, Info = SA::Info> + HasY<Y = SA::Y>,
    SA::Y: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.y().partial_cmp(&other.y())
    }
}

impl<SA,SB> Ord for Pair<SA, SB>
where
    Self: HasY<Y = SA::Y>,
    SA::Y: Ord,
    SA: Solution + HasY,
    SB: Solution<Id = SA::Id, Info = SA::Info> + HasY<Y = SA::Y>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.y().cmp(&other.y())
    }
}


impl<SA,SB> Orderable for Pair<SA, SB>
where
    Self: HasY<Y = SA::Y>,
    SA::Y: Orderable,
    SA: Solution + HasY,
    SB: Solution<Id = SA::Id, Info = SA::Info> + HasY<Y = SA::Y>,
{
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.y().ord_cmp(&other.y())
    }
}

impl<SA,SB> Dominate for Pair<SA, SB>
where
    Self: HasY<Y = SA::Y>,
    SA::Y: ElemMulti,
    SA: Solution + HasY + Dominate,
    SB: Solution<Id = SA::Id, Info = SA::Info> + HasY<Y = SA::Y> + Dominate,
{
    fn dominates(&self, other: &Self) -> bool {
        self.0.dominates(&other.0)
    }

    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.0.get_objective_by_index(idx)
    }

    fn len_objectives(&self) -> usize {
        self.0.len_objectives()
    }

    fn get_objectives(&self) -> &[f64] {
        self.0.get_objectives()
    }
}

//---------------------//
//--- LONE SOLUTION ---//
//---------------------//

/// A single [`Solution`].
///
/// This is used for objective-only searchspaces.
/// When the right-hand sides of the following declarations `[! name | ... | ... !]` in [`hpo!`](../../../tantale/macros/macro.hpo.html) or
/// [`objective!`](../../../tantale/macros/macro.objective.html) are empty, the created solutions are wrapped in a [`Lone`] instead of a [`Pair`].
/// This prevent unecessary computations and cloning.
///
/// # Example
/// ```
/// use tantale::core::{Uncomputed, BaseSol, EmptyInfo, HasId, Id, SolutionShape, Lone, Real, SId};
/// use std::sync::Arc;
///
/// let info = Arc::new(EmptyInfo {});
/// let obj = BaseSol::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.2]), info);
/// let lone = Lone::new(obj);
///
/// assert_eq!(lone.id(), lone.get_sobj().id());
/// ```
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "S: Solution")]
pub struct Lone<S: Solution>(S);

impl<S: Solution> Lone<S>
{
    /// Create a new lone solution wrapper.
    pub fn new(sol: S) -> Self {
        Self(sol)
    }
}

impl<S: Solution> Linked for Lone<S>
{
    type Obj = S::Dom;
    type Opt = S::Dom;
    type TrueOpt = NoDomain;
}

impl<S: Solution> HasId for Lone<S>
{
    type Id = S::Id;
    fn id(&self) -> S::Id {
        self.0.id()
    }

    fn ref_id(&self) -> &S::Id {
        self.0.ref_id()
    }

    fn mut_ref_id(&mut self) -> &mut S::Id {
        self.0.mut_ref_id()
    }
}

impl<S: Solution> HasStepId for Lone<S>
where
    S::Id: StepId,
{
    fn increment(&mut self) {
        self.0.id().increment();
    }

    fn id_step(&self) -> usize {
        self.0.id().id_step()
    }

    fn previous_id(&self) -> S::Id {
        self.0.id().previous_id()
    }
}

impl<S> HasInfo for Lone<S>
where
    S : Solution + HasInfo,
{
    type Info = S::Info;
    fn sinfo(&self) -> Arc<S::Info> {
        self.0.sinfo()
    }
}

impl<S> HasY for Lone<S>
where
    S: Solution + HasY,
{
    type Y = S::Y;
    fn y(&self) -> Arc<Self::Y> {
        self.0.y()
    }
}

impl<S> HasStep for Lone<S>
where
    S: Solution + HasStep,
{
    fn step(&self) -> Step {
        self.0.step()
    }

    fn raw_step(&self) -> EvalStep {
        self.0.raw_step()
    }

    fn pending(&mut self) {
        self.0.pending();
    }

    fn partially(&mut self, value: isize) {
        self.0.partially(value);
    }

    fn discard(&mut self) {
        self.0.discard();
    }

    fn evaluated(&mut self) {
        self.0.evaluated();
    }

    fn error(&mut self) {
        self.0.error();
    }

    fn set_raw_step(&mut self, value: EvalStep) {
        self.0.set_raw_step(value);
    }
}

impl<S> HasFidelity for Lone<S>
where
    S: Solution + HasFidelity,
{
    fn fidelity(&self) -> Fidelity {
        self.0.fidelity()
    }

    fn set_fidelity(&mut self, fidelity: f64) {
        self.0.set_fidelity(fidelity);
    }
}

impl<S: Solution> SolutionShape for Lone<S>
{
    type ObjSol = S;
    type OptSol = S;

    fn get_sobj(&self) -> &S {
        &self.0
    }
    fn get_sopt(&self) -> &S {
        &self.0
    }
    fn get_mut_sobj(&mut self) -> &mut S {
        &mut self.0
    }
    fn get_mut_sopt(&mut self) -> &mut S {
        &mut self.0
    }
    fn extract_sobj(self) -> S {
        self.0
    }
    fn extract_sopt(self) -> S {
        self.0
    }
    fn _clone_shape(&self) -> Self {
        let obj = self.get_sobj()._clone_sol();
        Self::new(obj)
    }
}

impl<S:Uncomputed> UncomputedShape for Lone<S>
{
}

impl<S: Uncomputed> IntoComputedShape for Lone<S>
{
    type ComputedShape<Out:Outcome> = Lone<Computed<S, Out>>;
    
    fn into_computed<Out:Outcome>(self, y: Arc<TypeCodom<Out>>) -> Self::ComputedShape<Out> {
        Lone::new(Computed::new(self.0, y))
    }
    
    fn extract<Out:Outcome>(comp: Self::ComputedShape<Out>) -> (Self, Arc<TypeCodom<Out>>) {
        let y = comp.0.y;
        let x = comp.0.sol;
        (Lone::new(x), y)
    }
}

impl<S> PartialEq for Lone<S>
where
    Self: HasY<Y = S::Y>,
    S: Solution + HasY,
    S::Y: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.y() == other.y()
    }
}

impl<S> Eq for Lone<S>
where
    Self: HasY<Y = S::Y>,
    S: Solution + HasY,
    S::Y: Eq,
{
}

impl<S> PartialOrd for Lone<S>
where
    Self: HasY<Y = S::Y>,
    S: Solution + HasY,
    S::Y: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.y().partial_cmp(&other.y())
    }
}

impl<S> Ord for Lone<S>
where
    Self: HasY<Y = S::Y>,
    S: Solution + HasY,
    S::Y: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.y().cmp(&other.y())
    }
}

impl<S> Orderable for Lone<S>
where
    Self: HasY<Y = S::Y>,
    S: Solution + HasY,
    S::Y: Orderable,
{
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.y().ord_cmp(&other.y())
    }
}

impl<S: Solution> Dominate for Lone<S>
where
    Self: HasY<Y = S::Y>,
    S::Y: ElemMulti,
    S: Solution + HasY + Dominate,
{
    fn dominates(&self, other: &Self) -> bool {
        self.y().dominates(&other.y())
    }

    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.y().get_objective_by_index(idx)
    }

    fn len_objectives(&self) -> usize {
        self.y().len_objectives()
    }

    fn get_objectives(&self) -> &[f64] {
        self.0.get_objectives()
    }
}
