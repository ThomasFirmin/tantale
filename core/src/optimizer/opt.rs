use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    searchspace::Searchspace,
    solution::{Computed, Partial, SolInfo},
};
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

/// Describes information linked to a group of [`Solutions`](Solution)
/// obtained  after each iteration of the [`Optimizer`].
pub trait OptInfo {}

/// An empty [`OptInfo`] or [`SolInfo`].
pub struct EmptyInfo {}
impl SolInfo for EmptyInfo {}
impl OptInfo for EmptyInfo {}

pub type ArcVecArc<T> = Arc<Vec<Arc<T>>>;

/// The [`Optimizer`] is one of the elemental software brick of the library.
/// It describes how to sample [`Solutions`](Solution) in order to **maximize**
/// the [`Objective`] function.
pub trait Optimizer<PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>
where
    PObj: Partial<Obj, SInfo>,
    CObj: Computed<PObj, Obj, SInfo, Cod, Out>,
    POpt: Partial<Opt, SInfo>,
    COpt: Computed<POpt, Opt, SInfo, Cod, Out>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<PObj, POpt, Obj, Opt, SInfo>,
    Info: OptInfo,
{
    /// Initialize the [`Optimizer`]
    fn init(&mut self);

    /// Executed once at the beginning of the optimization. Does not require previous [`Computed`].
    fn first_step(&mut self, sp: &Scp, pid: u32) -> (ArcVecArc<PObj>, ArcVecArc<POpt>, Info);

    /// Computes a single iteration of the [`Optimizer`]. It must return a slice of [`Solution`]`<Opt,Cod, Out, SInfo, DIM>`
    /// and some optimizer info [`OptInfo`]. [`Self`] is mutable in order to update the [`Optimizer`]'s state.
    /// Requires previously [`Computed`] `x` [`Solution`].
    fn step(
        &mut self,
        x: (ArcVecArc<CObj>, ArcVecArc<COpt>),
        sp: &Scp,
        pid: u32,
    ) -> (ArcVecArc<PObj>, ArcVecArc<POpt>, Info);
}

#[cfg(feature = "par")]
pub trait ParallelOptimizer<PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>:
    Optimizer<PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>
where
    PObj: Partial<Obj, SInfo>,
    CObj: Computed<PObj, Obj, SInfo, Cod, Out>,
    POpt: Partial<Opt, SInfo>,
    COpt: Computed<POpt, Opt, SInfo, Cod, Out>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<PObj, POpt, Obj, Opt, SInfo>,
    Info: OptInfo,
{
    fn interact(&self);
    fn update(&self);
}
