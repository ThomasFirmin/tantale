use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    saver::CSVWritable,
    searchspace::Searchspace,
    solution::{Computed, Id, ParSId, Partial, SolInfo},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
/// Describes information linked to a group of [`Solutions`](Solution)
/// obtained  after each iteration of the [`Optimizer`].
pub trait OptInfo
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}

/// Describes the current state of an [`Optimizer`].
/// It is used to serialize and deserialize the [`Optimizer`].
pub trait OptState
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}

/// An empty [`OptInfo`] or [`SolInfo`].
#[derive(Serialize, Deserialize, std::fmt::Debug)]
pub struct EmptyInfo {}
impl SolInfo for EmptyInfo {}
impl OptInfo for EmptyInfo {}
impl CSVWritable<(), ()> for EmptyInfo {
    fn header(_elem: &()) -> Vec<String> {
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
    Arc<Info>,
);

/// The [`Optimizer`] is one of the elemental software brick of the library.
/// It describes how to sample [`Solutions`](Solution) in order to **maximize**
/// the [`Objective`] function.
pub trait Optimizer<SolId, Obj, Opt, Cod, Out, Scp>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId, Obj, Opt, Self::SInfo>,
{
    type SInfo: SolInfo;
    type Info: OptInfo;
    type State: OptState;
    /// Initialize the [`Optimizer`]
    fn init(&mut self);

    /// Executed once at the beginning of the optimization. Does not require previous [`Computed`].
    fn first_step(&mut self, sp: Arc<Scp>) -> OptOutput<SolId, Obj, Opt, Self::SInfo, Self::Info>;

    /// Computes a single iteration of the [`Optimizer`]. It must return a slice of [`Solution`]`<Opt,Cod, Out, SInfo, DIM>`
    /// and some optimizer info [`OptInfo`]. [`Self`] is mutable in order to update the [`Optimizer`]'s state.
    /// Requires previously [`Computed`] `x` [`Solution`].
    fn step(
        &mut self,
        x: SolPairs<SolId, Obj, Opt, Cod, Out, Self::SInfo>,
        sp: Arc<Scp>,
    ) -> OptOutput<SolId, Obj, Opt, Self::SInfo, Self::Info>;

    /// Returns the current [`OptState`] of the [`Optimizer`].
    fn get_state(&mut self) -> &Self::State;

    /// Return an instance of the [`Optimizer`]  from an [`OptState`].
    fn from_state(state: Self::State) -> Self;
}

/// A sequential [`Optimizer`] without any parallelization.
pub trait SequentialOptimizer<SolId, Obj, Opt, Cod, Out, Scp>:
    Optimizer<SolId, Obj, Opt, Cod, Out, Scp>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId, Obj, Opt, Self::SInfo>,
{
}

/// A parallel [`Optimizer`] with multi-processing.
pub trait ParallelOptimizer<Obj, Opt, Cod, Out, Scp>:
    Optimizer<ParSId, Obj, Opt, Cod, Out, Scp>
where
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<ParSId, Obj, Opt, Self::SInfo>,
{
    fn interact(&self);
    fn update(&self);
}
