use crate::{
    domain::Domain,
    objective::{Codomain, FuncWrapper, Outcome},
    recorder::csv::CSVWritable,
    searchspace::Searchspace,
    solution::{BatchType, Id, SolInfo},
    Onto, Partial,
};

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub type OpInfType<Op, SolId, Obj, Opt, Out, Scp, Fn> =
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::Info;
pub type OpSInfType<Op, SolId, Obj, Opt, Out, Scp, Fn> =
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::SInfo;
pub type OpCodType<Op, SolId, Obj, Opt, Out, Scp, Fn> =
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::Cod;
pub type OpSolType<Op, SolId, Obj, Opt, Out, Scp, Fn> =
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::Sol;

/// Computed [`BatchType`], the associated type of a [`BatchType`] knwowing the optimizer.
pub type PBType<Op, SolId, Obj, Opt, Out, Scp, Fn> =
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::BType;
pub type CBType<Op, SolId, Obj, Opt, Out, Scp, Fn> =
    <<Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::BType as BatchType<
        SolId,
        Obj,
        Opt,
        <Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::SInfo,
        <Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::Sol,
        <Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::Info,
    >>::Comp<<Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::Cod, Out>;
pub type OBType<Op, SolId, Obj, Opt, Out, Scp, Fn> =
    <<Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::BType as BatchType<
        SolId,
        Obj,
        Opt,
        <Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::SInfo,
        <Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::Sol,
        <Op as Optimizer<SolId, Obj, Opt, Out, Scp, Fn>>::Info,
    >>::Outc<Out>;

/// Describes the type of iteration:
/// * Monothreaded: The evaluation of a [`BatchType`] is done within a single thread.
/// * Threaded: The evaluation of a [`BatchType`] is multi-threaded.
/// * Distributed: The evaluation of a [`BatchType`] is MPI-distributed.
///
/// # Notes
///
/// According to the [`BatchType`] and algorithm type, the parallelization of the iteration level might be synchronous, i.e. all the [`BatchType`] is evaluated
/// before the next [`step`](Optimizer::step), or asynchronous, i.e. [`BatchType`] are generated on the fly / on demand.
#[derive(Serialize, Deserialize)]
pub enum IterMode {
    Monothreaded,
    Threaded,
    Distributed,
}

/// Describes the type of the optimizer execution:
/// * Mono: A single instance of the algorithm is executed.
/// * Threaded: Multiple instances of the optimizer are executed within different threads, and can interact with eachothers ([`MultiInstanceOptimizer`]).
/// * Distributed: Multiple instances of the optimizer are MPI-distributed, and can interact with eachothers ([`MultiInstanceOptimizer`]).
#[derive(Serialize, Deserialize)]
pub enum AlgoMode {
    Mono,
    Threaded,
    Distributed,
}

/// Describes information linked to a group of [`Solutions`](Solution)
/// obtained  after each iteration of the [`Optimizer`].
pub trait OptInfo
where
    Self: Serialize + for<'de> Deserialize<'de> + Debug,
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

/// The [`Optimizer`] is one of the elemental software brick of the library.
/// It describes how to sample [`Solutions`](Solution) in order to **maximize**
/// the [`Objective`] function.
pub trait Optimizer<SolId, Obj, Opt, Out, Scp, Fn>
where
    Self: Sized,
    SolId: Id,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom>,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom>,
    Out: Outcome,
    Scp: Searchspace<Self::Sol, SolId, Obj, Opt, Self::SInfo>,
    Fn: FuncWrapper,
    Self::Sol: Partial<SolId, Obj, Self::SInfo>,
    <Self::Sol as Partial<SolId, Obj, Self::SInfo>>::Twin<Opt>:
        Partial<SolId, Opt, Self::SInfo, Twin<Obj> = Self::Sol>,

{
    type Sol: Partial<SolId, Obj, Self::SInfo>;
    type BType: BatchType<SolId,Obj,Opt,Self::SInfo, Self::Sol, Self::Info>;
    type State: OptState;
    type Cod: Codomain<Out>;
    type SInfo: SolInfo;
    type Info: OptInfo;

    /// Initialize the [`Optimizer`]
    fn init(&mut self);

    /// Executed once at the beginning of the optimization. Does not require previous [`Computed`].
    fn first_step(&mut self, scp: &Scp) -> Self::BType;

    /// Computes a single iteration of the [`Optimizer`]. It must return a slice of [`Solution`]`<Opt,Cod, Out, SInfo, DIM>`
    /// and some optimizer info [`OptInfo`]. [`Self`] is mutable in order to update the [`Optimizer`]'s state.
    /// Requires previously [`Computed`] `x` [`Solution`].
    fn step(&mut self, x: CBType<Self,SolId,Obj,Opt,Out,Scp,Fn>, scp: &Scp) -> Self::BType;

    /// Returns the current [`OptState`] of the [`Optimizer`].
    fn get_state(&mut self) -> &Self::State;

    /// Return an instance of the [`Optimizer`]  from an [`OptState`].
    fn from_state(state: Self::State) -> Self;
}

/// A parallel [`Optimizer`] with multi-processing.
pub trait MultiInstanceOptimizer<SolId, Obj, Opt, Out, Scp, Fn>:
    Optimizer<SolId, Obj, Opt, Out, Scp, Fn>
where
    Self: Sized,
    SolId: Id,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom>,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom>,
    Out: Outcome,
    Scp: Searchspace<Self::Sol, SolId, Obj, Opt, Self::SInfo>,
    Fn: FuncWrapper,
    Self::Sol: Partial<SolId, Obj, Self::SInfo>,
    <Self::Sol as Partial<SolId, Obj, Self::SInfo>>::Twin<Opt>:
        Partial<SolId, Opt, Self::SInfo, Twin<Obj> = Self::Sol>,
{
    fn interact(&self);
    fn update(&self);
}
