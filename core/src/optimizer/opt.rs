use crate::{
    Stop,
    domain::Domain,
    experiment::Runable,
    objective::{Codomain, FuncWrapper, Outcome},
    saver::{CSVWritable, Saver},
    searchspace::Searchspace,
    solution::{BatchType, Id, SolInfo}
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug,sync::Arc};

pub type OpInfType<Op,SolId,Obj,Opt,Out,Scp> = <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Info;
pub type OpSInfType<Op,SolId,Obj,Opt,Out,Scp> = <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::SInfo;
pub type OpCodType<Op,SolId,Obj,Opt,Out,Scp> = <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Cod;
pub type OpBatchType<Op,SolId,Obj,Opt,Out,Scp> = <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::BType;

/// Computed [`BatchType`], the associated type of a [`BatchType`] knwowing the optimizer.
pub type PBType<Op, SolId, Obj, Opt, Out, Scp> = <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::BType;
pub type CBType<Op, SolId, Obj, Opt, Out, Scp> = <<Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::BType as BatchType<SolId,Obj,Opt,<Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::SInfo,<Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Info>>::Comp<<Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Cod,Out>;
pub type OBType<Op, SolId, Obj, Opt, Out, Scp> = <<Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::BType as BatchType<SolId,Obj,Opt,<Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::SInfo,<Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Info>>::Outc<Out>;


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
    Distributed
}

/// Describes the type of the optimizer execution:
/// * Monothreaded: A single instance of the algorithm is executed.
/// * Threaded: Multiple instances of the optimizer are executed within different threads, and can interact with eachothers ([`MultiInstanceOptimizer`]).
/// * Distributed: Multiple instances of the optimizer are MPI-distributed, and can interact with eachothers ([`MultiInstanceOptimizer`]).
#[derive(Serialize, Deserialize)]
pub enum AlgoMode {
    Monothreaded,
    Threaded,
    Distributed
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
    /// Set the iteration level type of parallelism of the [`Optimizer`]. See [`IterMode`].
    /// By default an [`Optimizer`] is set to [`MonoThreaded`](IterMode::MonoThreaded).
    fn set_iter_lvl(&mut self, mode: IterMode);

    /// Get the iteration level type of parallelism [`Optimizer`]. See [`IterMode`].
    /// By default an [`Optimizer`] is set to [`MonoThreaded`](IterMode::MonoThreaded).
    fn get_iter_lvl(&self) -> &IterMode;    
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
pub type VecArc<T> = Vec<Arc<T>>;

/// The [`Optimizer`] is one of the elemental software brick of the library.
/// It describes how to sample [`Solutions`](Solution) in order to **maximize**
/// the [`Objective`] function.
pub trait Optimizer<SolId, Obj, Opt, Out, Scp>
where
    Self: Sized,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<SolId, Obj, Opt, Self::SInfo>,
{
    type SInfo: SolInfo;
    type Info: OptInfo;
    type State: OptState;
    type BType: BatchType<SolId,Obj,Opt,Self::SInfo,Self::Info>;
    type FnWrap: FuncWrapper;
    type Cod: Codomain<Out>;

    /// Initialize the [`Optimizer`]
    fn init(&mut self);

    /// Executed once at the beginning of the optimization. Does not require previous [`Computed`].
    fn first_step(&mut self, sp: Arc<Scp>) -> Self::BType;

    /// Computes a single iteration of the [`Optimizer`]. It must return a slice of [`Solution`]`<Opt,Cod, Out, SInfo, DIM>`
    /// and some optimizer info [`OptInfo`]. [`Self`] is mutable in order to update the [`Optimizer`]'s state.
    /// Requires previously [`Computed`] `x` [`Solution`].
    fn step(
        &mut self,
        x: CBType<Self,SolId,Obj,Opt,Out,Scp>,
        sp: Arc<Scp>,
    ) ->Self::BType;

    /// Returns the current [`OptState`] of the [`Optimizer`].
    fn get_state(&mut self) -> &Self::State;

    /// Return an instance of the [`Optimizer`]  from an [`OptState`].
    fn from_state(state: Self::State) -> Self;

    fn to_exp<Run,St,Sv>(self, searchspace: Scp, objective: Self::FnWrap, stop: St, saver:Sv)->Run
    where
        Run : Runable<SolId, Scp, Self, St, Sv, Out, Obj, Opt>,
        St: Stop,
        Sv: Saver<SolId, St, Obj, Opt, Out, Scp, Self, Run::Eval>,
    ;
}

/// A parallel [`Optimizer`] with multi-processing.
pub trait MultiInstanceOptimizer<SolId, Obj, Opt, Out, Scp>:
    Optimizer<SolId, Obj, Opt, Out, Scp>
where
    Self: Sized,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<SolId, Obj, Opt, Self::SInfo>,
{   
    fn set_algo_lvl(&mut self, mode: AlgoMode);
    fn interact(&self);
    fn update(&self);
}