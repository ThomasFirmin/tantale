use crate::{
    Searchspace,
    checkpointer::Checkpointer,
    domain::onto::LinkOpt,
    objective::{FuncWrapper, Outcome},
    optimizer::{Optimizer, opt::{BatchOptimizer, ObjRaw, SequentialOptimizer}},
    recorder::Recorder,
    searchspace::CompShape,
    solution::{Batch, Id, OutBatch, SolutionShape},
    stop::Stop
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::DistCheckpointer, experiment::mpi::{utils::{MPIProcess, SendRec, XMsg}, worker::Worker}, recorder::DistRecorder, solution::{HasY, shape::SolObj}
};

// SYNCHRONOUS
pub mod synchronous;
pub use synchronous::evaluator::{BatchEvaluator, ThrBatchEvaluator};
pub use synchronous::fidevaluator::{FidBatchEvaluator, FidThrBatchEvaluator};
pub use synchronous::syncrun::{MonoExperiment, ThrExperiment};

#[cfg(feature = "mpi")]
pub mod mpi;
#[cfg(feature = "mpi")]
pub use synchronous::syncrun::DistExperiment;

#[macro_export]
#[cfg(not(feature = "mpi"))]
macro_rules! experiment {
    (($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::MonoExperiment::new(
            ($domain, $codomain),
            $objective,
            $optimizer,
            $stop,
            ($rec, $check),
        )
    };
    (Mono, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::MonoExperiment::new(
            ($domain, $codomain),
            $objective,
            $optimizer,
            $stop,
            ($rec, $check),
        )
    };
    (Threaded, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::ThrExperiment::new(
            ($domain, $codomain),
            $objective,
            $optimizer,
            $stop,
            ($rec, $check),
        )
    };
}

#[macro_export]
#[cfg(feature = "mpi")]
macro_rules! experiment {
    (($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::MonoExperiment<_,_,_,_,_,_,_,_,_,_,_> as 
            tantale::core::experiment::Runable<_,_,_,_,_,_,_,_,_,_>>::new(
            ($domain, $codomain),
            $objective,
            $optimizer,
            $stop,
            ($rec, $check),
        )
    };
    (Mono, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::MonoExperiment<_,_,_,_,_,_,_,_,_,_,_> as 
            tantale::core::experiment::Runable<_,_,_,_,_,_,_,_,_,_>>::new(
            ($domain, $codomain),
            $objective,
            $optimizer,
            $stop,
            ($rec, $check),
        )
    };
    (Threaded, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::ThrExperiment<_,_,_,_,_,_,_,_,_,_,_> as 
            tantale::core::experiment::Runable<_,_,_,_,_,_,_,_,_,_>>::new(
            ($domain, $codomain),
            $objective,
            $optimizer,
            $stop,
            ($rec, $check),
        )
    };
    (Distributed, $proc:expr, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::DistExperiment<_,_,_,_,_,_,_,_,_,_,_,_> as 
            tantale::core::experiment::DistRunable<_,_,_,_,_,_,_,_,_,_>>::new(
            $proc,
            ($domain, $codomain),
            $objective,
            $optimizer,
            $stop,
            ($rec, $check),
        )
    };
}

#[macro_export]
#[cfg(not(feature = "mpi"))]
macro_rules! load {
    (($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::MonoExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_,_>::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Mono, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::MonoExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_,_>::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Threaded, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::ThrExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_,_>::load(($domain,$codomain), $objective, ($rec, $check))
    };
}

#[macro_export]
#[cfg(feature = "mpi")]
macro_rules! load {
    (($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::MonoExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_,_> as
            tantale::core::experiment::Runable<_,_,_,_,_,_,_,_,_,_>
            >::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Mono, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::MonoExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_,_> as
            tantale::core::experiment::Runable<_,_,_,_,_,_,_,_,_,_>
            >::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Threaded, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::ThrExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_,_> as
            tantale::core::experiment::Runable<_,_,_,_,_,_,_,_,_,_>
            >::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Distributed, $proc:expr, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::DistExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_,_,_> as
            tantale::core::experiment::DistRunable<_,_,_,_,_,_,_,_,_,_>
            >::load($proc, ($domain,$codomain), $objective, ($rec, $check))
    };
}

/// [`Runable`] describes the optimization loop.
/// The [`Optimizer`] defines a single iteration, and a [`Runable`] loops over an [`Optimizer`] step.
/// It wraps-up, the [`Optimizer`], the stopping criterion [`Stop`], the [`Objective`], the [`Recorder`] and [`Checkpointer`].
pub trait Runable<SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    St: Stop,
    Rec: Recorder<SolId,Out,Scp,Op>,
    Check: Checkpointer,
    Out: Outcome,
    Fn: FuncWrapper<ObjRaw<Op, Scp, SolId, Out>>,
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
}

#[cfg(feature = "mpi")]
pub enum MasterWorker<'a, DRun, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    DRun: DistRunable<'a, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    St: Stop,
    Rec: DistRecorder<SolId,Out,Scp,Op>,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<ObjRaw<Op, Scp, SolId, Out>>,
{
    Master(DRun),
    Worker(DRun::WType),
}

#[cfg(feature = "mpi")]
impl<'a, DRun, SolId, Scp, Op, St, Rec, Check, Out, Fn>
    MasterWorker<'a, DRun, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    DRun: DistRunable<'a, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    St: Stop,
    Rec: DistRecorder<SolId,Out,Scp,Op>,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<ObjRaw<Op, Scp, SolId, Out>>,
{
    pub fn run(self) {
        match self {
            MasterWorker::Master(exp) => exp.run(),
            MasterWorker::Worker(worker) => worker.run(),
        }
    }
}

#[cfg(feature = "mpi")]
/// [`DistRunable`] describes a MPI-distributed [`Runable`], defined by a [`MasterWorker`] parallelization.
pub trait DistRunable<'a, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    Self:Sized,
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    St: Stop,
    Rec: DistRecorder<SolId,Out,Scp,Op>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasY<Op::Cod,Out>,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<ObjRaw<Op, Scp, SolId, Out>>,
{
    type WType: Worker<SolId>;
    fn new(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Fn,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> MasterWorker<'a, Self, SolId, Scp, Op, St, Rec, Check, Out, Fn>;
    fn run(self);
    fn load(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Fn,
        saver: (Option<Rec>, Check),
    ) -> MasterWorker<'a, Self, SolId, Scp, Op, St, Rec, Check, Out, Fn>;
}

//------------------------//
//-------EVALUATOR--------//
//------------------------//

pub trait Evaluate
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}

/// [`SingleEvaluate`] is an [`Evaluate`] describing how to evaluate the output of a sequential [`Optimizer`]
/// generating a **single** [`Partial`] at each step.
pub trait SingleEvaluate<SolId, Op, Scp, Out, St, Fn>:Evaluate
where
    SolId:Id,
    Op: SequentialOptimizer<SolId,LinkOpt<Scp>,Out,Scp,Fn>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    Out:Outcome,
    St:Stop,
    Fn: FuncWrapper<ObjRaw<Op, Scp, SolId, Out>>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: &Fn,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> (Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,OutBatch<SolId,Op::Info,Out>);
    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::SolShape>);
}

/// [`MonoEvaluate`] is an [`Evaluate`] describing how to evaluate the output of a sequential [`Optimizer`]
/// generating a **batch** of [`Partial`] at each step.
pub trait MonoEvaluate<SolId, Op, Scp, Out, St, Fn>: Evaluate
where
    SolId:Id,
    Op: BatchOptimizer<SolId,LinkOpt<Scp>,Out,Scp,Fn>,
    Scp: Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    Out: Outcome,
    St: Stop,
    Fn: FuncWrapper<ObjRaw<Op, Scp, SolId, Out>>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: &Fn,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> (Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,OutBatch<SolId,Op::Info,Out>);
    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::SolShape>);
}

/// [`ThrEvaluate`] is an [`Evaluate`] describing how to evaluate, with multi-threading, the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait ThrEvaluate<SolId, Op, Scp, Out, St, Fn>:Evaluate
where
    SolId:Id,
    Op: BatchOptimizer<SolId,LinkOpt<Scp>,Out,Scp,Fn>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    Out:Outcome,
    St:Stop,
    Fn: FuncWrapper<ObjRaw<Op, Scp, SolId, Out>>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Fn>,
        cod: Arc<Op::Cod>,
        stop: Arc<Mutex<St>>,
    ) -> (Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,OutBatch<SolId,Op::Info,Out>);
    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::SolShape>);
}

#[cfg(feature = "mpi")]
/// [`DistEvaluate`] is an [`Evaluate`] describing how to distribute, with MPI, the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait DistEvaluate<SolId, Op, Scp, Out, St, Fn, M>:Evaluate
where
    SolId:Id,
    Op: BatchOptimizer<SolId,LinkOpt<Scp>,Out,Scp,Fn>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasY<Op::Cod,Out>,
    Out:Outcome,
    St:Stop,
    Fn: FuncWrapper<ObjRaw<Op, Scp, SolId, Out>>,
    M:XMsg<SolObj<Scp::SolShape,SolId,Op::SInfo>,SolId,Scp::Obj,Op::SInfo>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        sendrec: &mut SendRec<'_,M,Scp::SolShape,SolId,Op::SInfo,Op::Cod,Out>,
        ob: &Fn,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> (Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,OutBatch<SolId,Op::Info,Out>);
    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::SolShape>);
}

// pub use mpi::synchronous::mpifidseqrun::{FidExperiment,FidEvaluator};
// pub use mpi::synchronous::mpifidthrrun::{FidThrExperiment,FidThrEvaluator};
