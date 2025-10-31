use crate::{
    Onto, Partial, Stop,
    recorder::{Recorder, csv::CSVWritable},
    checkpointer::Checkpointer,
    domain::Domain,
    experiment::{MonoEvaluate, Runable, ThrEvaluate},
    objective::{Codomain, FuncWrapper, Outcome},
    searchspace::Searchspace,
    solution::{BatchType, Id, SolInfo}
};
#[cfg(feature = "mpi")]
use crate::{
    checkpointer::DistCheckpointer,
    experiment::{DistEvaluate, DistRunable,
        mpi::tools::MPIProcess
    },
    recorder::DistRecorder
};

use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};

pub type OpInfType<Op, SolId, Obj, Opt, Out, Scp> =
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Info;
pub type OpSInfType<Op, SolId, Obj, Opt, Out, Scp> =
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::SInfo;
pub type OpCodType<Op, SolId, Obj, Opt, Out, Scp> =
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Cod;
pub type OpBatchType<Op, SolId, Obj, Opt, Out, Scp> =
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::BType;
pub type OpSolType<Op, SolId, Obj, Opt, Out, Scp> =
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Sol;

/// Computed [`BatchType`], the associated type of a [`BatchType`] knwowing the optimizer.
pub type PBType<Op, SolId, Obj, Opt, Out, Scp> =
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::BType;
pub type CBType<Op, SolId, Obj, Opt, Out, Scp> =
    <<Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::BType as BatchType<
        SolId,
        Obj,
        Opt,
        <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::SInfo,
        <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Info,
    >>::Comp<<Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Cod, Out>;
pub type OBType<Op, SolId, Obj, Opt, Out, Scp> =
    <<Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::BType as BatchType<
        SolId,
        Obj,
        Opt,
        <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::SInfo,
        <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Info,
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

pub type ArcVecArc<T> = Arc<Vec<Arc<T>>>;
pub type VecArc<T> = Vec<Arc<T>>;


/// The [`Optimizer`] is one of the elemental software brick of the library.
/// It describes how to sample [`Solutions`](Solution) in order to **maximize**
/// the [`Objective`] function.
pub trait Optimizer<SolId, Obj, Opt, Out, Scp>
where
    Self: Sized,
    SolId: Id,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom>,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom>,
    Out: Outcome,
    Scp: Searchspace<Self::Sol, SolId, Obj, Opt, Self::SInfo>,
    Self::Sol: Partial<SolId, Obj, Self::SInfo>,
    <Self::Sol as Partial<SolId, Obj, Self::SInfo>>::Twin<Opt>:
        Partial<SolId, Opt, Self::SInfo, Twin<Obj> = Self::Sol>,
{
    type Sol: Partial<SolId, Obj, Self::SInfo>;
    type BType: BatchType<SolId, Obj, Opt, Self::SInfo, Self::Info>;
    type State: OptState;
    type FnWrap: FuncWrapper;
    type Cod: Codomain<Out>;
    type SInfo: SolInfo;
    type Info: OptInfo;

    /// Initialize the [`Optimizer`]
    fn init(&mut self);

    /// Executed once at the beginning of the optimization. Does not require previous [`Computed`].
    fn first_step(&mut self, sp: Arc<Scp>) -> Self::BType;

    /// Computes a single iteration of the [`Optimizer`]. It must return a slice of [`Solution`]`<Opt,Cod, Out, SInfo, DIM>`
    /// and some optimizer info [`OptInfo`]. [`Self`] is mutable in order to update the [`Optimizer`]'s state.
    /// Requires previously [`Computed`] `x` [`Solution`].
    fn step(&mut self, x: CBType<Self, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>) -> Self::BType;

    /// Returns the current [`OptState`] of the [`Optimizer`].
    fn get_state(&mut self) -> &Self::State;

    /// Return an instance of the [`Optimizer`]  from an [`OptState`].
    fn from_state(state: Self::State) -> Self;
}

pub trait MonoOptimizer<SolId, Obj, Opt, Out, Scp>: Optimizer<SolId, Obj, Opt, Out, Scp>
where
    Self: Sized,
    SolId: Id,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom>,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom>,
    Out: Outcome,
    Scp: Searchspace<Self::Sol, SolId, Obj, Opt, Self::SInfo>,
    Self::Sol: Partial<SolId, Obj, Self::SInfo>,
    <Self::Sol as Partial<SolId, Obj, Self::SInfo>>::Twin<Opt>:
        Partial<SolId, Opt, Self::SInfo, Twin<Obj> = Self::Sol>,
{
    type Eval<St: Stop>: MonoEvaluate<Self, St, Obj, Opt, Out, SolId, Scp>;
    type Exp<St, Rec, Check>: Runable<Self::Eval<St>, SolId, Scp, Self, St, Rec, Check, Out, Obj, Opt>
    where
        St: Stop,
        Rec: Recorder<SolId,Obj,Opt,Out,Scp,Self>,
        Check: Checkpointer<SolId,St,Obj,Opt,Out,Scp,Self,Self::Eval<St>>;
    fn get_mono<St, Rec, Check>(
        searchspace: Scp,
        objective: Self::FnWrap,
        optimizer: Self,
        stop: St,
        recorder: Option<Rec>,
        checkpointer: Option<Check>,
    ) -> Self::Exp<St, Rec, Check>
    where
        St: Stop,
        Rec: Recorder<SolId,Obj,Opt,Out,Scp,Self>,
        Check: Checkpointer<SolId,St,Obj,Opt,Out,Scp,Self,Self::Eval<St>>,
    {
        Self::Exp::new(searchspace, objective, optimizer, stop, recorder, checkpointer)
    }
    fn load_mono<St, Rec, Check>(
        searchspace: Scp,
        objective: Self::FnWrap,
        recorder: Option<Rec>,
        checkpointer: Option<Check>
    ) -> Self::Exp<St, Rec, Check>
    where
        St: Stop,
        Rec: Recorder<SolId,Obj,Opt,Out,Scp,Self>,
        Check: Checkpointer<SolId,St,Obj,Opt,Out,Scp,Self,Self::Eval<St>>,
    {
        <Self::Exp<St,Rec,Check> as Runable<Self::Eval<St>,SolId,Scp,Self,St,Rec,Check,Out,Obj,Opt>>::load(
            searchspace,
            objective,
            recorder,
            checkpointer,
        )
    }
}

pub trait ThrOptimizer<SolId, Obj, Opt, Out, Scp>: Optimizer<SolId, Obj, Opt, Out, Scp>
where
    Self: Sized,
    SolId: Id + Send + Sync,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom> + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom> + Send + Sync,
    Opt::TypeDom: Send + Sync,
    Out: Outcome + Send + Sync,
    Scp: Searchspace<Self::Sol, SolId, Obj, Opt, Self::SInfo> + Send + Sync,
    Self::Sol: Partial<SolId, Obj, Self::SInfo> + Send + Sync,
    <Self::Sol as Partial<SolId, Obj, Self::SInfo>>::Twin<Opt>:
        Partial<SolId, Opt, Self::SInfo, Twin<Obj> = Self::Sol>,
{
    type Eval<St: Stop + Send + Sync>: ThrEvaluate<Self, St, Obj, Opt, Out, SolId, Scp>;
    type Exp<St, Rec, Check>: Runable<Self::Eval<St>, SolId, Scp, Self, St, Rec, Check, Out, Obj, Opt>
    where
        St: Stop + Send + Sync,
        Rec: Recorder<SolId,Obj,Opt,Out,Scp,Self> + Send + Sync,
        Check: Checkpointer<SolId,St,Obj,Opt,Out,Scp,Self,Self::Eval<St>> + Send + Sync;
    fn get_threaded<St, Rec, Check>(
        searchspace: Scp,
        objective: Self::FnWrap,
        optimizer: Self,
        stop: St,
        recorder: Rec,
        checkpointer: Check,
    ) -> Self::Exp<St, Rec, Check>
    where
        St: Stop + Send + Sync,
        Rec: Recorder<SolId,Obj,Opt,Out,Scp,Self> + Send + Sync,
        Check: Checkpointer<SolId,St,Obj,Opt,Out,Scp,Self,Self::Eval<St>> + Send + Sync,
    {
        Self::Exp::new(searchspace, objective, optimizer, stop, recorder, checkpointer)
    }

    fn load_threaded<St, Rec, Check>(
        searchspace: Scp,
        objective: Self::FnWrap,
        recorder: Rec,
        checkpointer: Check,
    ) -> Self::Exp<St, Rec, Check>
    where
        St: Stop + Send + Sync,
        Rec: Recorder<SolId,Obj,Opt,Out,Scp,Self> + Send + Sync,
        Check: Checkpointer<SolId,St,Obj,Opt,Out,Scp,Self,Self::Eval<St>> + Send + Sync,
    {
        <Self::Exp<St,Rec,Check> as Runable<Self::Eval<St>,SolId,Scp,Self,St,Rec,Check,Out,Obj,Opt>>::load(
            searchspace,
            objective,
            recorder,
            checkpointer,
        )
    }
}

#[cfg(feature = "mpi")]
pub trait DistOptimizer<SolId, Obj, Opt, Out, Scp>: Optimizer<SolId, Obj, Opt, Out, Scp>
where
    Self: Sized,
    SolId: Id,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom>,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom>,
    Out: Outcome,
    Scp: Searchspace<Self::Sol, SolId, Obj, Opt, Self::SInfo>,
    Self::Sol: Partial<SolId, Obj, Self::SInfo>,
    <Self::Sol as Partial<SolId, Obj, Self::SInfo>>::Twin<Opt>:
        Partial<SolId, Opt, Self::SInfo, Twin<Obj> = Self::Sol>,
{
    type Eval<St: Stop>: DistEvaluate<Self, St, Obj, Opt, Out, SolId, Scp>;
    type Exp<St, Rec, Check>: DistRunable<Self::Eval<St>, SolId, Scp, Self, St, Rec, Check, Out, Obj, Opt>
    where
        St: Stop + Send + Sync,
        Rec: DistRecorder<SolId,Obj,Opt,Out,Scp,Self> + Send + Sync,
        Check: DistCheckpointer<SolId,St,Obj,Opt,Out,Scp,Self,Self::Eval<St>> + Send + Sync;

    fn get_distributed<St, Rec, Check>(
        proc: &MPIProcess,
        searchspace: Scp,
        objective: Self::FnWrap,
        optimizer: Self,
        stop: St,
        recorder: Rec,
        checkpointer: Check,
    ) -> Self::Exp<St, Rec, Check>
    where
        St: Stop + Send + Sync,
        Rec: DistRecorder<SolId,Obj,Opt,Out,Scp,Self> + Send + Sync,
        Check: DistCheckpointer<SolId,St,Obj,Opt,Out,Scp,Self,Self::Eval<St>> + Send + Sync,
    {
        Self::Exp::new_dist(proc, searchspace, objective, optimizer, stop, recorder, checkpointer)
    }

    fn load_distributed<St, Rec, Check>(
        proc: &MPIProcess,
        searchspace: Scp,
        objective: Self::FnWrap,
        recorder: Rec,
        checkpointer: Check,
    ) -> Self::Exp<St, Rec, Check>
    where
        St: Stop + Send + Sync,
        Rec: DistRecorder<SolId,Obj,Opt,Out,Scp,Self> + Send + Sync,
        Check: DistCheckpointer<SolId,St,Obj,Opt,Out,Scp,Self,Self::Eval<St>> + Send + Sync,
    {
        <Self::Exp<St, Rec, Check> as DistRunable<
            Self::Eval<St>,
            SolId,
            Scp,
            Self,
            St,
            Rec,
            Check,
            Out,
            Obj,
            Opt,
        >>::load_dist(proc,objective,recorder,checkpointer)
    }
}

/// A parallel [`Optimizer`] with multi-processing.
pub trait MultiInstanceOptimizer<SolId, Obj, Opt, Out, Scp>:
    Optimizer<SolId, Obj, Opt, Out, Scp>
where
    Self: Sized,
    SolId: Id,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom>,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom>,
    Out: Outcome,
    Scp: Searchspace<Self::Sol, SolId, Obj, Opt, Self::SInfo>,
    Self::Sol: Partial<SolId, Obj, Self::SInfo>,
    <Self::Sol as Partial<SolId, Obj, Self::SInfo>>::Twin<Opt>:
        Partial<SolId, Opt, Self::SInfo, Twin<Obj> = Self::Sol>,
{
    fn interact(&self);
    fn update(&self);
}
