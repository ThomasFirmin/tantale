use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    saver::CSVWritable,
    searchspace::Searchspace,
    solution::{Partial, Computed, Id, ParSId, SId, SolInfo},
};
use std::{
    sync::Arc,
};
use serde::{Serialize,Deserialize};
/// Describes information linked to a group of [`Solutions`](Solution)
/// obtained  after each iteration of the [`Optimizer`].
pub trait OptInfo {}

/// Describes the current state of an [`Optimizer`].
/// It is used to serialize and deserialize the [`Optimizer`].
pub trait OptState {}

/// An empty [`OptInfo`] or [`SolInfo`].
#[derive(Serialize,Deserialize,std::fmt::Debug)]
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

/// [`Arc`] [`Vec`] of paired `Obj` and `Opt`[`Computed`]
pub type SolPairs<SolId, ADom, BDom, Cod, Out, SInfo> = (
    ArcVecArc<Computed<SolId, ADom, Cod, Out, SInfo>>,
    ArcVecArc<Computed<SolId, BDom, Cod, Out, SInfo>>,
);
/// Output of an [`Optimizer`] made of [`Arc`] [`Vec`] of paired `Obj` and `Opt`[`Partial`] and an [`OptInfo`].
pub type OptOutput<SolId, ADom, BDom, SInfo, Info> = (
    ArcVecArc<Partial<SolId, ADom, SInfo>>,
    ArcVecArc<Partial<SolId, BDom, SInfo>>,
    Arc<Info>
);

/// The [`Optimizer`] is one of the elemental software brick of the library.
/// It describes how to sample [`Solutions`](Solution) in order to **maximize**
/// the [`Objective`] function.
pub trait Optimizer<SolId,Obj, Opt, SInfo, Cod, Out, Scp, Info, State>
where
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId,Obj, Opt, SInfo>,
    Info: OptInfo,
    SolId: Id + PartialEq + Clone + Copy,
    State: OptState,
{
    /// Initialize the [`Optimizer`]
    fn init(&mut self);

    /// Executed once at the beginning of the optimization. Does not require previous [`Computed`].
    fn first_step(&mut self, sp: Arc<Scp>) -> OptOutput<SolId, Obj, Opt, SInfo, Info>;

    /// Computes a single iteration of the [`Optimizer`]. It must return a slice of [`Solution`]`<Opt,Cod, Out, SInfo, DIM>`
    /// and some optimizer info [`OptInfo`]. [`Self`] is mutable in order to update the [`Optimizer`]'s state.
    /// Requires previously [`Computed`] `x` [`Solution`].
    fn step(
        &mut self,
        x: SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
        sp: Arc<Scp>,
    ) -> OptOutput<SolId, Obj, Opt, SInfo, Info>;

    /// Returns the current [`OptState`] of the [`Optimizer`].
    fn get_state(&mut self) -> &State;
}

/// A sequential [`Optimizer`] without any parallelization.
pub trait SequentialOptimizer<Obj, Opt, SInfo, Cod, Out, Scp, Info, State>:
    Optimizer<SId, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>
where
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SId, Obj, Opt, SInfo>,
    Info: OptInfo,
    State: OptState,
{
}

/// A parallel [`Optimizer`] with multi-processing.
pub trait ParallelOptimizer<Obj, Opt, SInfo, Cod, Out, Scp, Info, State>:
    Optimizer<ParSId, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>
where
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<ParSId, Obj, Opt, SInfo>,
    Info: OptInfo,
    State: OptState,
{
    fn interact(&self);
    fn update(&self);
}
