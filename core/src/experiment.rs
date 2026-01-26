use crate::{
    SId, Searchspace, checkpointer::{Checkpointer, ThrCheckpointer}, domain::onto::LinkOpt, objective::{FuncWrapper, Outcome}, optimizer::Optimizer, recorder::Recorder, searchspace::CompShape, solution::{Batch, Id, OutBatch, SolutionShape, Uncomputed, shape::RawObj}, stop::Stop
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use ::mpi::Rank;
#[cfg(feature = "mpi")]
use crate::{
    checkpointer::DistCheckpointer,
    experiment::mpi::{
        utils::{MPIProcess, SendRec, XMsg},
        worker::Worker,
    },
    recorder::DistRecorder,
    solution::{shape::SolObj, HasY},
};

// BASICS
pub mod basics;
pub use basics::{MonoExperiment,ThrExperiment};
// BATCHED
pub mod batched;
pub use batched::batchevaluator::{BatchEvaluator, ThrBatchEvaluator};
pub use batched::batchfidevaluator::{FidBatchEvaluator, FidThrBatchEvaluator};
// SEQUENTIAL
pub mod sequential;

#[cfg(feature = "mpi")]
pub mod mpi;
#[cfg(feature = "mpi")]
pub use basics::MPIExperiment;

/// Returns a [`MonoExperiment`].
pub fn mono<PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
    space: (Scp, Op::Cod),
    objective: Fn,
    optimizer: Op,
    stop: St,
    saver: (Option<Rec>, Option<Check>),
) -> impl Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    MonoExperiment<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>,
    Op: Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SId, Op::SInfo>,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>: SolutionShape<SId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: Checkpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SId, Op::SInfo>>,
    Eval: Evaluate,
{
    <MonoExperiment<_,_,_,_,_,_,_,_,_,_> as Runable<_,_,_,_,_,_,_,_,_>>::new(
        space, objective, optimizer, stop, saver
    )
}

/// Returns a [`ThrExperiment`].
pub fn threaded<PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
    space: (Scp, Op::Cod),
    objective: Fn,
    optimizer: Op,
    stop: St,
    saver: (Option<Rec>, Option<Check>),
) -> impl Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    ThrExperiment<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>,
    Op: Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SId, Op::SInfo>,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>: SolutionShape<SId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: ThrCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SId, Op::SInfo>>,
    Eval: Evaluate,
{
    <ThrExperiment<_,_,_,_,_,_,_,_,_,_> as Runable<_,_,_,_,_,_,_,_,_>>::new(
        space, objective, optimizer, stop, saver
    )
}

#[cfg(feature = "mpi")]
#[allow(clippy::type_complexity)]
/// Returns a [`MPIExperiment`].
pub fn distributed<'a, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
    proc: &'a MPIProcess,
    space: (Scp, Op::Cod),
    objective: Fn,
    optimizer: Op,
    stop: St,
    saver: (Option<Rec>, Option<Check>),
) -> MasterWorker<'a, MPIExperiment<'a, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    MPIExperiment<'a, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        MPIRunable<'a, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>,
    Op: Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SId, Op::SInfo>,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>: SolutionShape<SId, Op::SInfo>,
    St: Stop,
    Rec: DistRecorder<PSol, SId, Out, Scp, Op>,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SId, Op::SInfo>>,
    Eval: Evaluate,
{
    <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::new(
        proc,
        space, objective, optimizer, stop, saver
    )
}

/// Load a [`MonoExperiment`] from a saver (recorder optional, checkpointer required).
pub fn mono_load<Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
    space: (Scp, Op::Cod),
    objective: Fn,
    saver: (Option<Rec>, Check),
) -> impl Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>
where
    Op: Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>,
    St: Stop,
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    MonoExperiment<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>,
    Scp: Searchspace<PSol, SId, Op::SInfo>,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo>,
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: Checkpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SId, Op::SInfo>>,
    Eval: Evaluate,
{
    <MonoExperiment<_,_,_,_,_,_,_,_,_,_> as Runable<_,_,_,_,_,_,_,_,_>>::load(
        space, objective, saver
    )
}

/// Load a [`ThrExperiment`] from a saver (recorder optional, threaded checkpointer required).
pub fn threaded_load<Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
    space: (Scp, Op::Cod),
    objective: Fn,
    saver: (Option<Rec>, Check),
) -> impl Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>
where
    Op: Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>,
    St: Stop,
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    ThrExperiment<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>,
    Scp: Searchspace<PSol, SId, Op::SInfo>,
    CompShape<Scp, PSol, SId, Op::SInfo, <Op as Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>>::Cod, Out>:
        SolutionShape<SId, Op::SInfo>,
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: ThrCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SId, Op::SInfo>>,
    Eval: Evaluate,
{
    <ThrExperiment<_,_,_,_,_,_,_,_,_,_> as Runable<_,_,_,_,_,_,_,_,_>>::load(
        space, objective, saver
    )
}

#[cfg(feature = "mpi")]
#[allow(clippy::type_complexity)]
/// Load a [`MPIExperiment`] from a saver (dist-recorder optional, dist-checkpointer required).
pub fn distributed_load<'a, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
    proc: &'a MPIProcess,
    space: (Scp, <Op as Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>>::Cod),
    objective: Fn,
    saver: (Option<Rec>, Check),
) -> MasterWorker<'a, MPIExperiment<'a, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>
where
    Op: Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>,
    St: Stop,
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    MPIExperiment<'a, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        MPIRunable<'a, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>,
    Scp: Searchspace<PSol, SId, Op::SInfo>,
    CompShape<Scp, PSol, SId, Op::SInfo, <Op as Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>>::Cod, Out>:
        SolutionShape<SId, Op::SInfo>,
    Rec: DistRecorder<PSol, SId, Out, Scp, Op>,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SId, Op::SInfo>>,
    Eval: Evaluate,
{
    <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::load(
        proc,
        space,
        objective,
        saver
    )
}


/// Helper macro to call the `_load` helpers while only specifying the `Op` and `St` types.
/// Usage:
/// - load!(mono, MyOp, MyStop, space, objective, saver)
/// - load!(threaded, MyOp, MyStop, space, objective, saver)
/// - load!(distributed, proc, MyOp, MyStop, space, objective, saver)  // requires "mpi" feature
#[macro_export]
#[cfg(not(feature = "mpi"))]
macro_rules! load {
    (mono, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::mono_load::<$Op, $St>($space, $objective, $saver)
    };
    (threaded, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::threaded_load::<$Op, $St>($space, $objective, $saver)
    };
    (distributed, $proc:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::distributed_load::<$Op, $St>($proc, $space, $objective, $saver)
    };
}

/// Helper macro to call the `_load` helpers while only specifying the `Op` and `St` types.
/// Usage:
/// - load!(mono, MyOp, MyStop, space, objective, saver)
/// - load!(threaded, MyOp, MyStop, space, objective, saver)
/// - load!(distributed, proc, MyOp, MyStop, space, objective, saver)  // requires "mpi" feature
#[macro_export]
#[cfg(feature = "mpi")]
macro_rules! load {
    (mono, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::mono_load::<$Op, $St,_,_,_,_,_,_,_>($space, $objective, $saver)
    };
    (threaded, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::threaded_load::<$Op, $St,_,_,_,_,_,_,_>($space, $objective, $saver)
    };
    (distributed, $proc:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::distributed_load::<$Op, $St,_,_,_,_,_,_,_>($proc, $space, $objective, $saver)
    };
}

/// [`Runable`] describes the optimization loop.
/// The [`Optimizer`] defines a single iteration, and a [`Runable`] loops over an [`Optimizer`] step.
/// It wraps-up, the [`Optimizer`], the stopping criterion [`Stop`], the [`Objective`], the [`Recorder`] and [`Checkpointer`].
pub trait Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SolId, Out, Scp, Op>,
    Check: Checkpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Fn,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> Self;
    fn run(self);
    fn load(space: (Scp, Op::Cod), objective: Fn, saver: (Option<Rec>, Check)) -> Self;
    fn get_stop(&self) -> &St;
    fn get_searchspace(&self) -> &Scp;
    fn get_codomain(&self) -> &Op::Cod;
    fn get_objective(&self) -> &Fn;
    fn get_optimizer(&self) -> &Op;
    fn get_recorder(&self) -> Option<&Rec>;
    fn get_checkpointer(&self) -> Option<&Check>;
    fn get_mut_stop(&mut self) -> &mut St;
    fn get_mut_searchspace(&mut self) -> &mut Scp;
    fn get_mut_codomain(&mut self) -> &mut Op::Cod;
    fn get_mut_objective(&mut self) -> &mut Fn;
    fn get_mut_optimizer(&mut self) -> &mut Op;
    fn get_mut_recorder(&mut self) -> Option<&mut Rec>;
    fn get_mut_checkpointer(&mut self) -> Option<&mut Check>;
}

#[cfg(feature = "mpi")]
pub enum MasterWorker<'a, DRun, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    DRun: MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    St: Stop,
    Rec: DistRecorder<PSol, SolId, Out, Scp, Op>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    Master(DRun),
    Worker(DRun::WType),
}

#[cfg(feature = "mpi")]
impl<'a, DRun, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
    MasterWorker<'a, DRun, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    DRun: MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Rec: DistRecorder<PSol, SolId, Out, Scp, Op>,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    pub fn run(self) {
        match self {
            MasterWorker::Master(exp) => exp.run(),
            MasterWorker::Worker(worker) => worker.run(),
        }
    }
}

#[cfg(feature = "mpi")]
/// [`MPIRunable`] describes a MPI-distributed [`Runable`], defined by a [`MasterWorker`] parallelization.
pub trait MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    Self: Sized,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    St: Stop,
    Rec: DistRecorder<PSol, SolId, Out, Scp, Op>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    type WType: Worker<SolId>;
    fn new(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Fn,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> MasterWorker<'a, Self, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>;
    fn run(self);
    fn load(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Fn,
        saver: (Option<Rec>, Check),
    ) -> MasterWorker<'a, Self, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>;
    fn get_stop(&self) -> &St;
    fn get_searchspace(&self) -> &Scp;
    fn get_codomain(&self) -> &Op::Cod;
    fn get_objective(&self) -> &Fn;
    fn get_optimizer(&self) -> &Op;
    fn get_recorder(&self) -> Option<&Rec>;
    fn get_checkpointer(&self) -> Option<&Check>;
    fn get_mut_stop(&mut self) -> &mut St;
    fn get_mut_searchspace(&mut self) -> &mut Scp;
    fn get_mut_codomain(&mut self) -> &mut Op::Cod;
    fn get_mut_objective(&mut self) -> &mut Fn;
    fn get_mut_optimizer(&mut self) -> &mut Op;
    fn get_mut_recorder(&mut self) -> Option<&mut Rec>;
    fn get_mut_checkpointer(&mut self) -> Option<&mut Check>;
}

//------------------------//
//-------EVALUATOR--------//
//------------------------//

pub type OutBatchEvaluate<SolId, SInfo, Info, Scp, PSol, Cod, Out> = (
    Batch<SolId, SInfo, Info, CompShape<Scp, PSol, SolId, SInfo, Cod, Out>>,
    OutBatch<SolId, Info, Out>,
);

pub type OutShapeEvaluate<SolId, SInfo, Scp, PSol, Cod, Out> = (
    CompShape<Scp, PSol, SolId, SInfo, Cod, Out>,
    (SolId,Out)
);
#[cfg(feature="mpi")]
pub type DistOutShapeEvaluate<SolId, SInfo, Scp, PSol, Cod, Out> = (
    Rank,
    (CompShape<Scp, PSol, SolId, SInfo, Cod, Out>,
    (SolId,Out))
);

pub trait Evaluate
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}

/// [`MonoEvaluate`] is an [`Evaluate`] describing how to evaluate the output of a [`Optimizer`].
pub trait MonoEvaluate<PSol, SolId, Op, Scp, Out, St, Fn, OutType>: Evaluate
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    Out: Outcome,
    St: Stop,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: &Fn,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> OutType;
}

/// [`ThrEvaluate`] is an [`Evaluate`] describing how to evaluate, with multi-threading, the output of a [`Optimizer`].
pub trait ThrEvaluate<PSol, SolId, Op, Scp, Out, St, Fn,OutType>: Evaluate
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    Out: Outcome,
    St: Stop,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Fn>,
        cod: Arc<Op::Cod>,
        stop: Arc<Mutex<St>>,
    ) -> OutType;
}

#[cfg(feature = "mpi")]
/// [`DistEvaluate`] is an [`Evaluate`] describing how to distribute, with MPI, the output of a [`Optimizer`].
pub trait DistEvaluate<PSol, SolId, Op, Scp, Out, St, Fn, M, OutType>: Evaluate
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    Out: Outcome,
    St: Stop,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    M: XMsg<SolObj<Scp::SolShape, SolId, Op::SInfo>, SolId, Scp::Obj, Op::SInfo>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        sendrec: &mut SendRec<'_, M, Scp::SolShape, SolId, Op::SInfo, Op::Cod, Out>,
        ob: &Fn,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> OutType;
}

// pub use mpi::synchronous::mpifidseqrun::{FidExperiment,FidEvaluator};
// pub use mpi::synchronous::mpifidthrrun::{FidThrExperiment,FidThrEvaluator};
