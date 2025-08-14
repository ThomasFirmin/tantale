use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    saver::CSVWritable,
    searchspace::Searchspace,
    solution::{Computed, Id, ParSId, Partial, SId, SolInfo},
};
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

/// Describes information linked to a group of [`Solutions`](Solution)
/// obtained  after each iteration of the [`Optimizer`].
pub trait OptInfo {}

/// Describes the current state of an [`Optimizer`].
/// It is used to serialize and deserialize the [`Optimizer`].
pub trait OptState {}

/// An empty [`OptInfo`] or [`SolInfo`].
pub struct EmptyInfo {}
impl SolInfo for EmptyInfo {}
impl OptInfo for EmptyInfo {}
impl CSVWritable<()> for EmptyInfo {
    fn header(&self) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::new()
    }
}

pub type ArcVecArc<T> = Arc<Vec<Arc<T>>>;

pub type SolPairs<SolId, A, ADom, B, BDom, Cod, Out, SInfo> = (
    ArcVecArc<Computed<SolId, A, ADom, Cod, Out, SInfo>>,
    ArcVecArc<Computed<SolId, B, BDom, Cod, Out, SInfo>>,
);

/// The [`Optimizer`] is one of the elemental software brick of the library.
/// It describes how to sample [`Solutions`](Solution) in order to **maximize**
/// the [`Objective`] function.
pub trait Optimizer<SolId, PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>
where
    PObj: Partial<SolId, Obj, SInfo>,
    POpt: Partial<SolId, Opt, SInfo>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId, PObj, POpt, Obj, Opt, SInfo>,
    Info: OptInfo,
    SolId: Id + PartialEq + Clone + Copy,
    State: OptState,
{
    /// Initialize the [`Optimizer`]
    fn init(&mut self);

    /// Executed once at the beginning of the optimization. Does not require previous [`Computed`].
    fn first_step(&mut self, sp: Arc<Scp>) -> (ArcVecArc<PObj>, ArcVecArc<POpt>, Info);

    /// Computes a single iteration of the [`Optimizer`]. It must return a slice of [`Solution`]`<Opt,Cod, Out, SInfo, DIM>`
    /// and some optimizer info [`OptInfo`]. [`Self`] is mutable in order to update the [`Optimizer`]'s state.
    /// Requires previously [`Computed`] `x` [`Solution`].
    fn step(
        &mut self,
        x: SolPairs<SolId, PObj, Obj, POpt, Opt, Cod, Out, SInfo>,
        sp: Arc<Scp>,
    ) -> (ArcVecArc<PObj>, ArcVecArc<POpt>, Info);

    /// Returns the current [`OptState`] of the [`Optimizer`].
    fn get_state(&mut self) -> &State;
}

/// A sequential [`Optimizer`] without any parallelization.
pub trait SequentialOptimizer<PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>:
    Optimizer<SId, PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>
where
    PObj: Partial<SId, Obj, SInfo>,
    POpt: Partial<SId, Opt, SInfo>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SId, PObj, POpt, Obj, Opt, SInfo>,
    Info: OptInfo,
    State: OptState,
{
}

/// A parallel [`Optimizer`] with multi-processing.
pub trait ParallelOptimizer<PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>:
    Optimizer<ParSId, PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>
where
    PObj: Partial<ParSId, Obj, SInfo>,
    POpt: Partial<ParSId, Opt, SInfo>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<ParSId, PObj, POpt, Obj, Opt, SInfo>,
    Info: OptInfo,
    State: OptState,
{
    fn interact(&self);
    fn update(&self);
}
