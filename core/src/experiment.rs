use crate::{
    Domain, Partial, Searchspace, checkpointer::Checkpointer, domain::onto::{LinkObj, LinkOpt}, objective::{FuncWrapper, Outcome}, optimizer::{Optimizer, opt::OptCompBatch}, recorder::Recorder, solution::{Batch, Id, IntoComputed, OutBatch}, stop::Stop
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::DistCheckpointer,
    experiment::mpi::{utils::{MPIProcess, SendRec, XMsg}, worker::Worker},
    recorder::DistRecorder
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
pub trait Runable<SolId, Scp, Op, St, Rec, Check, Out, Opt, Fn>
where
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
    St: Stop,
    Rec: Recorder<SolId,Out,Scp,Op>,
    Check: Checkpointer,
    Opt: Domain,
    Out: Outcome,
    Fn: FuncWrapper,
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
pub enum MasterWorker<'a, DRun, SolId, Scp, Op, St, Rec, Check, Out, Opt, Fn>
where
    DRun: DistRunable<'a, SolId, Scp, Op, St, Rec, Check, Out, Opt, Fn>,
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
    St: Stop,
    Rec: DistRecorder<SolId,Out,Scp,Op>,
    Check: DistCheckpointer,
    Opt: Domain,
    Out: Outcome,
    Fn: FuncWrapper,
{
    Master(DRun),
    Worker(DRun::WType),
}

#[cfg(feature = "mpi")]
impl<'a, DRun, SolId, Scp, Op, St, Rec, Check, Out, Opt, Fn>
    MasterWorker<'a, DRun, SolId, Scp, Op, St, Rec, Check, Out, Opt, Fn>
where
    DRun: DistRunable<'a, SolId, Scp, Op, St, Rec, Check, Out, Opt, Fn>,
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
    St: Stop,
    Rec: DistRecorder<SolId,Out,Scp,Op>,
    Check: DistCheckpointer,
    Opt: Domain,
    Out: Outcome,
    Fn: FuncWrapper,
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
pub trait DistRunable<'a, SolId, Scp, Op, St, Rec, Check, Out, Opt, Fn>
where
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
    St: Stop,
    Rec: DistRecorder<SolId,Out,Scp,Op>,
    Check: DistCheckpointer,
    Opt: Domain,
    Out: Outcome,
    Fn: FuncWrapper,
{
    type WType: Worker<SolId>;
    fn new(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Fn,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> MasterWorker<'a, Self, SolId, Scp, Op, St, Rec, Check, Out, Opt, Fn>;
    fn run(self);
    fn load(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Fn,
        saver: (Option<Rec>, Check),
    ) -> MasterWorker<'a, Self, SolId, Scp, Op, St, Rec, Check, Out, Opt, Fn>;
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
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
    Out:Outcome,
    St:Stop,
    Fn: FuncWrapper,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Fn>,
        stop: Arc<Mutex<St>>,
    ) -> (OptCompBatch<Op,Scp,SolId,Out>,OutBatch<SolId,Op::Info,Out>);
    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::PartShape>);
}

/// [`MonoEvaluate`] is an [`Evaluate`] describing how to evaluate the output of a sequential [`Optimizer`]
/// generating a **batch** of [`Partial`] at each step.
pub trait MonoEvaluate<SolId, Op, Scp, Out, St, Fn>: Evaluate
where
    SolId:Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
    Out:Outcome,
    St:Stop,
    Fn: FuncWrapper,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Fn>,
        stop: Arc<Mutex<St>>,
    ) -> (OptCompBatch<SolId,Op,Scp,Out>,OutBatch<SolId,Op::Info,Out>);
    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::PartShape>);
}

/// [`ThrEvaluate`] is an [`Evaluate`] describing how to evaluate, with multi-threading, the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait ThrEvaluate<SolId, Op, Scp, Out, St, Fn>:Evaluate
where
    SolId:Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
    Out:Outcome,
    St:Stop,
    Fn: FuncWrapper,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Fn>,
        stop: Arc<Mutex<St>>,
    ) -> (OptCompBatch<SolId,Op,Scp,Out>,OutBatch<SolId,Op::Info,Out>);
    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::PartShape>);
}

#[cfg(feature = "mpi")]
/// [`DistEvaluate`] is an [`Evaluate`] describing how to distribute, with MPI, the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait DistEvaluate<SolId, Op, Scp, Out, St, Fn, M>:Evaluate
where
    SolId:Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
    Out:Outcome,
    St:Stop,
    Fn: FuncWrapper,
    M:XMsg<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>,SolId,Scp::Obj,Op::SInfo>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        sendrec: &mut SendRec<'_,M,Scp::PartShape,SolId,Op::SInfo,Op::Cod,Out>,
        ob: &Fn,
        stop: &mut St,
    ) -> (OptCompBatch<SolId,Op,Scp,Out>,OutBatch<SolId,Op::Info,Out>);
    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::PartShape>);
}

// pub use mpi::synchronous::mpifidseqrun::{FidExperiment,FidEvaluator};
// pub use mpi::synchronous::mpifidthrrun::{FidThrExperiment,FidThrEvaluator};
