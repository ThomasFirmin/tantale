use crate::{
    domain::{onto::LinkOpt, Codomain, Domain},
    objective::{FuncWrapper, Outcome},
    recorder::csv::CSVWritable,
    searchspace::{CompShape, Searchspace},
    solution::{shape::RawObj, Batch, Id, IntoComputed, SolInfo, SolutionShape, Uncomputed},
};

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Describes the type of iteration:
/// * Monothreaded: The evaluation of a [`Batch`] is done within a single thread.
/// * Threaded: The evaluation of a [`Batch`] is multi-threaded.
/// * Distributed: The evaluation of a [`Batch`] is MPI-distributed.
///
/// # Notes
///
/// According to the [`Batch`] and algorithm type, the parallelization of the iteration level might be synchronous, i.e. all the [`Batch`] is evaluated
/// before the next [`step`](Optimizer::step), or asynchronous, i.e. [`Batch`] are generated on the fly / on demand.
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
    Self: Serialize + for<'de> Deserialize<'de> + Debug + Default,
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
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct EmptyInfo;
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

pub type OpInfType<Op, PSol, Scp, SolId, Out> =
    <Op as Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>>::Info;
pub type OpSInfType<Op, PSol, Scp, SolId, Out> =
    <Op as Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>>::SInfo;
pub type OpCodType<Op, PSol, Scp, SolId, Out> =
    <Op as Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>>::Cod;

/// The [`Optimizer`] is one of the elemental software brick of the library.
/// It describes how to sample [`Solutions`](Solution) in order to **maximize**
/// the [`Objective`] function.
pub trait Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
{
    type State: OptState;
    type Cod: Codomain<Out>;
    type SInfo: SolInfo;
    type Info: OptInfo;

    /// Returns the current [`OptState`] of the [`Optimizer`].
    fn get_state(&mut self) -> &Self::State;
    /// Return an instance of the [`Optimizer`]  from an [`OptState`].
    fn from_state(state: Self::State) -> Self;
}

/// A [`Batch`] of [`CompShape`].
pub type CompBatch<SolId, SInfo, Info, Scp, PSol, Cod, Out> =
    Batch<SolId, SInfo, Info, CompShape<Scp, PSol, SolId, SInfo, Cod, Out>>;

/// An [`Optimizer`] for which, at each step, a
pub trait BatchOptimizer<PSol, SolId, Opt, Out, Scp, Fn>:
    Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>: SolutionShape<SolId, Self::SInfo>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Self::SInfo>>,
{
    /// Executed once at the beginning of the optimization. Does not require previous [`Computed`].
    fn first_step(&mut self, scp: &Scp) -> Batch<SolId, Self::SInfo, Self::Info, Scp::SolShape>;
    /// Computes a single iteration of the [`Optimizer`]. It must return a slice of [`Solution`]`<Opt,Cod, Out, SInfo, DIM>`
    /// and some optimizer info [`OptInfo`]. [`Self`] is mutable in order to update the [`Optimizer`]'s state.
    /// Requires previously [`Computed`] `x` [`Solution`].
    fn step(
        &mut self,
        x: CompBatch<SolId, Self::SInfo, Self::Info, Scp, PSol, Self::Cod, Out>,
        scp: &Scp,
    ) -> Batch<SolId, Self::SInfo, Self::Info, Scp::SolShape>;
}

pub trait SequentialOptimizer<PSol, SolId, Opt, Out, Scp, Fn>:
    Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Self::SInfo>>,
{
    /// Executed once at the beginning of the optimization. Does not require previous [`Computed`].
    fn first_step(&mut self, scp: &Scp) -> Scp::SolShape;
    /// Computes a single iteration of the [`Optimizer`]. It must return a slice of [`Solution`]`<Opt,Cod, Out, SInfo, DIM>`
    /// and some optimizer info [`OptInfo`]. [`Self`] is mutable in order to update the [`Optimizer`]'s state.
    /// Requires previously [`Computed`] `x` [`Solution`].
    fn step(
        &mut self,
        x: CompShape<Scp, PSol, SolId, Self::SInfo, Self::Cod, Out>,
        scp: &Scp,
    ) -> Scp::SolShape;
}

/// A parallel [`Optimizer`] with multi-processing.
pub trait MultiInstanceOptimizer<PSol, SolId, Opt, Out, Scp, Fn>:
    Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Self::SInfo>>,
{
    fn interact(&self);
    fn update(&self);
}
