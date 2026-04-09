//! Solution shapes linking twins.
//!
//! A [`SolutionShape`] links two solutions with potentially different underlying domains or
//! types. The two solutions are [`twin`](crate::solution::Solution::twin)s and share the same [`Id`](crate::Id), letting users
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
    Codomain, Computed, EvalStep, Fidelity, Id, Outcome, SolInfo, Solution, StepId, domain::{
        Domain,
        onto::{LinkObj, LinkOpt, Linked},
    }, objective::Step, solution::{HasFidelity, HasId, HasSolInfo, HasStep, HasStepId, HasY, IntoComputed, Uncomputed}
};

use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData, sync::Arc};

/// Objective-side solution type for a [`SolutionShape`].
pub type SolObj<SolShape, SolId, SInfo> = <SolShape as SolutionShape<SolId, SInfo>>::SolObj;
/// Optimizer-side solution type for a [`SolutionShape`].
pub type SolOpt<SolShape, SolId, SInfo> = <SolShape as SolutionShape<SolId, SInfo>>::SolOpt;
/// Raw objective-side representation for a [`SolutionShape`].
pub type RawObj<SolShape, SolId, SInfo> =
    <<SolShape as SolutionShape<SolId, SInfo>>::SolObj as Solution<
        SolId,
        LinkObj<SolShape>,
        SInfo,
    >>::Raw;
/// Raw optimizer-side representation for a [`SolutionShape`].
pub type RawOpt<SolShape, SolId, SInfo> =
    <<SolShape as SolutionShape<SolId, SInfo>>::SolOpt as Solution<
        SolId,
        LinkOpt<SolShape>,
        SInfo,
    >>::Raw;

/// Trait linking objective- and optimizer-side solutions with a shared [`Id`](crate::Id).
pub trait SolutionShape<SolId: Id, SInfo: SolInfo>:
    Linked + HasId<SolId> + HasSolInfo<SInfo> + Debug + Sized
where
    Self: Serialize + for<'a> Deserialize<'a>,
{
    /// Objective-side solution type.
    type SolObj: Solution<SolId, Self::Obj, SInfo>;
    /// Optimizer-side solution type.
    type SolOpt: Solution<SolId, Self::Opt, SInfo>;

    /// Get a reference to the objective-side solution.
    fn get_sobj(&self) -> &Self::SolObj;
    /// Get a reference to the optimizer-side solution.
    fn get_sopt(&self) -> &Self::SolOpt;
    /// Get a mutable reference to the objective-side solution.
    fn get_mut_sobj(&mut self) -> &mut Self::SolObj;
    /// Get a mutable reference to the optimizer-side solution.
    fn get_mut_sopt(&mut self) -> &mut Self::SolOpt;
    /// Extract the objective-side solution, consuming the shape.
    fn extract_sobj(self) -> Self::SolObj;
    /// Extract the optimizer-side solution, consuming the shape.
    fn extract_sopt(self) -> Self::SolOpt;

    /// Creates a clone of this shape with the same solutions, metadata, and [`Id`], used
    /// for [`Accumulator`](crate::domain::codomain::Accumulator).
    fn _clone_shape(&self) -> Self;
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
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolObj: Serialize, SolOpt: Serialize",
    deserialize = "SolObj: for<'a> Deserialize<'a>, SolOpt: for<'a> Deserialize<'a>",
))]
pub struct Pair<SolObj, SolOpt, SolId, Obj, Opt, SInfo>(
    SolObj,
    SolOpt,
    PhantomData<SolId>,
    PhantomData<Obj>,
    PhantomData<Opt>,
    PhantomData<SInfo>,
)
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
    SolOpt: Solution<SolId, Opt, SInfo>;

/// Type alias for a computed [`Pair`] solution.
pub type CompPair<SolObj, SolOpt, SolId, Obj, Opt, SInfo, Cod, Out> = Pair<
    Computed<SolObj, SolId, Obj, Cod, Out, SInfo>,
    Computed<SolOpt, SolId, Opt, Cod, Out, SInfo>,
    SolId,
    Obj,
    Opt,
    SInfo,
>;

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo> Pair<SolObj, SolOpt, SolId, Obj, Opt, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
    SolOpt: Solution<SolId, Opt, SInfo>,
{
    /// Create a new paired solution from an objective and optimizer solution.
    pub fn new(solobj: SolObj, solopt: SolOpt) -> Self {
        Self(
            solobj,
            solopt,
            PhantomData,
            PhantomData,
            PhantomData,
            PhantomData,
        )
    }
}

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo> Linked for Pair<SolObj, SolOpt, SolId, Obj, Opt, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
    SolOpt: Solution<SolId, Opt, SInfo>,
{
    type Obj = Obj;
    type Opt = Opt;
}

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo> HasId<SolId>
    for Pair<SolObj, SolOpt, SolId, Obj, Opt, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
    SolOpt: Solution<SolId, Opt, SInfo>,
{
    fn id(&self) -> SolId {
        self.0.id()
    }
    
    fn ref_id(&self) -> &SolId {
        self.0.ref_id()
    }
    
    fn mut_ref_id(&mut self) -> &mut SolId {
        self.0.mut_ref_id()
    }
}

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo> HasStepId<SolId>
    for Pair<SolObj, SolOpt, SolId, Obj, Opt, SInfo>
where
    SolId: StepId,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasStepId<SolId> + HasStep,
    SolOpt: Solution<SolId, Opt, SInfo> + HasStepId<SolId> + HasStep,
{
    fn increment(&mut self) {
        self.0.increment();
        self.1.increment();
    }

    fn id_step(&self) -> usize {
        self.0.id_step()
    }

    fn previous_id(&self) -> SolId {
        self.0.previous_id()
    }
}

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo> HasSolInfo<SInfo>
    for Pair<SolObj, SolOpt, SolId, Obj, Opt, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
    SolOpt: Solution<SolId, Opt, SInfo>,
{
    fn sinfo(&self) -> Arc<SInfo> {
        self.0.sinfo()
    }
}

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo, Cod, Out> HasY<Cod, Out>
    for Pair<SolObj, SolOpt, SolId, Obj, Opt, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasY<Cod, Out>,
    SolOpt: Solution<SolId, Opt, SInfo> + HasY<Cod, Out>,
{
    fn y(&self) -> Arc<Cod::TypeCodom> {
        self.0.y()
    }
}

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo> HasStep
    for Pair<SolObj, SolOpt, SolId, Obj, Opt, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasStep,
    SolOpt: Solution<SolId, Opt, SInfo> + HasStep,
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

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo> HasFidelity
    for Pair<SolObj, SolOpt, SolId, Obj, Opt, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasFidelity,
    SolOpt: Solution<SolId, Opt, SInfo> + HasFidelity,
{
    fn fidelity(&self) -> Fidelity {
        self.0.fidelity()
    }

    fn set_fidelity(&mut self, fidelity: f64) {
        self.0.set_fidelity(fidelity);
        self.1.set_fidelity(fidelity);
    }
}

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo> SolutionShape<SolId, SInfo>
    for Pair<SolObj, SolOpt, SolId, Obj, Opt, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
    SolOpt: Solution<SolId, Opt, SInfo>,
{
    type SolObj = SolObj;
    type SolOpt = SolOpt;

    fn get_sobj(&self) -> &Self::SolObj {
        &self.0
    }
    fn get_sopt(&self) -> &Self::SolOpt {
        &self.1
    }
    fn get_mut_sobj(&mut self) -> &mut Self::SolObj {
        &mut self.0
    }
    fn get_mut_sopt(&mut self) -> &mut Self::SolOpt {
        &mut self.1
    }
    fn extract_sobj(self) -> Self::SolObj {
        self.0
    }
    fn extract_sopt(self) -> Self::SolOpt {
        self.1
    }
    fn _clone_shape(&self) -> Self {
        let obj = self.get_sobj()._clone_sol();
        let opt = self.get_sopt()._clone_sol();
        Self::new(obj, opt)
    }
}

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo> IntoComputed
    for Pair<SolObj, SolOpt, SolId, Obj, Opt, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolObj: Uncomputed<SolId, Obj, SInfo>,
    SolOpt: Uncomputed<SolId, Opt, SInfo>,
{
    type Computed<Cod: Codomain<Out>, Out: Outcome> = Pair<
        Computed<SolObj, SolId, Obj, Cod, Out, SInfo>,
        Computed<SolOpt, SolId, Opt, Cod, Out, SInfo>,
        SolId,
        Obj,
        Opt,
        SInfo,
    >;

    fn into_computed<Cod: Codomain<Out>, Out: Outcome>(
        self,
        y: Arc<Cod::TypeCodom>,
    ) -> Self::Computed<Cod, Out> {
        let cobj = Computed::new(self.0, y.clone());
        let copt = Computed::new(self.1, y);
        Pair::new(cobj, copt)
    }

    fn extract<Cod: Codomain<Out>, Out: Outcome>(
        comp: Self::Computed<Cod, Out>,
    ) -> (Self, Arc<Cod::TypeCodom>) {
        let y = comp.0.y;
        let pair = Pair::new(comp.0.sol, comp.1.sol);
        (pair, y)
    }
}

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo> From<(SolObj, SolOpt)>
    for Pair<SolObj, SolOpt, SolId, Obj, Opt, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
    SolOpt: Solution<SolId, Opt, SInfo>,
{
    fn from(value: (SolObj, SolOpt)) -> Self {
        Self(
            value.0,
            value.1,
            PhantomData,
            PhantomData,
            PhantomData,
            PhantomData,
        )
    }
}

//---------------------//
//--- LONE SOLUTION ---//
//---------------------//

/// A single [`Solution`] with no link to a [`Twin`](Solution::Twin).
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
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolObj: Serialize",
    deserialize = "SolObj: for<'a> Deserialize<'a>"
))]
pub struct Lone<SolObj, SolId, Obj, SInfo>(
    SolObj,
    PhantomData<SolId>,
    PhantomData<Obj>,
    PhantomData<SInfo>,
)
where
    SolId: Id,
    Obj: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>;

/// Type alias for a computed [`Lone`] solution.
pub type CompLone<SolObj, SolId, Obj, SInfo, Cod, Out> =
    Lone<Computed<SolObj, SolId, Obj, Cod, Out, SInfo>, SolId, Obj, SInfo>;

impl<SolObj, SolId, Obj, SInfo> Lone<SolObj, SolId, Obj, SInfo>
where
    SolId: Id,
    Obj: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
{
    /// Create a new lone solution wrapper.
    pub fn new(sol: SolObj) -> Self {
        Self(sol, PhantomData, PhantomData, PhantomData)
    }
}

impl<SolObj, SolId, Obj, SInfo> Linked for Lone<SolObj, SolId, Obj, SInfo>
where
    SolId: Id,
    Obj: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
{
    type Obj = Obj;
    type Opt = Obj;
}

impl<SolObj, SolId, Obj, SInfo> HasId<SolId> for Lone<SolObj, SolId, Obj, SInfo>
where
    SolId: Id,
    Obj: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasId<SolId>,
{
    fn id(&self) -> SolId {
        self.0.id()
    }
    
    fn ref_id(&self) -> &SolId {
        self.0.ref_id()
    }
    
    fn mut_ref_id(&mut self) -> &mut SolId {
        self.0.mut_ref_id()
    }
}

impl<SolObj, SolId, Obj, SInfo> HasStepId<SolId> for Lone<SolObj, SolId, Obj, SInfo>
where
    SolId: StepId,
    Obj: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasStepId<SolId> + HasStep,
{
    fn increment(&mut self) {
        self.0.increment();
    }

    fn id_step(&self) -> usize {
        self.0.id_step()
    }

    fn previous_id(&self) -> SolId {
        self.0.previous_id()
    }
}

impl<SolObj, SolId, Obj, SInfo> HasSolInfo<SInfo> for Lone<SolObj, SolId, Obj, SInfo>
where
    SolId: Id,
    Obj: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasSolInfo<SInfo>,
{
    fn sinfo(&self) -> Arc<SInfo> {
        self.0.sinfo()
    }
}

impl<SolObj, SolId, Obj, SInfo, Cod, Out> HasY<Cod, Out> for Lone<SolObj, SolId, Obj, SInfo>
where
    SolId: Id,
    Obj: Domain,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    SolObj: Solution<SolId, Obj, SInfo> + HasY<Cod, Out>,
{
    fn y(&self) -> Arc<Cod::TypeCodom> {
        self.0.y()
    }
}

impl<SolObj, SolId, Obj, SInfo> HasStep for Lone<SolObj, SolId, Obj, SInfo>
where
    SolId: Id,
    Obj: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasStep,
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

impl<SolObj, SolId, Obj, SInfo> HasFidelity for Lone<SolObj, SolId, Obj, SInfo>
where
    SolId: Id,
    Obj: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasFidelity,
{
    fn fidelity(&self) -> Fidelity {
        self.0.fidelity()
    }

    fn set_fidelity(&mut self, fidelity: f64) {
        self.0.set_fidelity(fidelity);
    }
}

impl<SolId, SInfo, Obj, SolObj> SolutionShape<SolId, SInfo> for Lone<SolObj, SolId, Obj, SInfo>
where
    SolId: Id,
    SInfo: SolInfo,
    Obj: Domain,
    SolObj: Solution<SolId, Obj, SInfo>,
{
    type SolObj = SolObj;
    type SolOpt = SolObj;

    fn get_sobj(&self) -> &Self::SolObj {
        &self.0
    }
    fn get_sopt(&self) -> &Self::SolOpt {
        &self.0
    }
    fn get_mut_sobj(&mut self) -> &mut Self::SolObj {
        &mut self.0
    }
    fn get_mut_sopt(&mut self) -> &mut Self::SolOpt {
        &mut self.0
    }
    fn extract_sobj(self) -> Self::SolObj {
        self.0
    }
    fn extract_sopt(self) -> Self::SolOpt {
        self.0
    }
    fn _clone_shape(&self) -> Self {
        let obj = self.get_sobj()._clone_sol();
        Self::new(obj)
    }
}

impl<SolObj, SolId, Obj, SInfo> IntoComputed for Lone<SolObj, SolId, Obj, SInfo>
where
    SolId: Id,
    Obj: Domain,
    SInfo: SolInfo,
    SolObj: Uncomputed<SolId, Obj, SInfo>,
{
    type Computed<Cod: Codomain<Out>, Out: Outcome> =
        Lone<Computed<SolObj, SolId, Obj, Cod, Out, SInfo>, SolId, Obj, SInfo>;

    fn into_computed<Cod: Codomain<Out>, Out: Outcome>(
        self,
        y: Arc<Cod::TypeCodom>,
    ) -> Self::Computed<Cod, Out> {
        Lone::new(Computed::new(self.0, y))
    }

    fn extract<Cod: Codomain<Out>, Out: Outcome>(
        comp: Self::Computed<Cod, Out>,
    ) -> (Self, Arc<Cod::TypeCodom>) {
        let y = comp.0.y;
        let pair = Lone::new(comp.0.sol);
        (pair, y)
    }
}

impl<SolObj, SolId, Obj, SInfo> From<SolObj> for Lone<SolObj, SolId, Obj, SInfo>
where
    SolId: Id,
    Obj: Domain,
    SInfo: SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
{
    fn from(value: SolObj) -> Self {
        Self(value, PhantomData, PhantomData, PhantomData)
    }
}
