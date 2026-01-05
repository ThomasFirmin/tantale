use crate::{
    checkpointer::Checkpointer,
    domain::onto::LinkOpt,
    objective::{FuncWrapper, Outcome},
    optimizer::{
        opt::{BatchOptimizer, SequentialOptimizer},
        Optimizer,
    },
    recorder::Recorder,
    searchspace::CompShape,
    solution::{shape::RawObj, Batch, Id, OutBatch, SolutionShape, Uncomputed},
    stop::Stop,
    Searchspace,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

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

// SYNCHRONOUS
pub mod batched;
pub use batched::evaluator::{BatchEvaluator, ThrBatchEvaluator};
pub use batched::fidevaluator::{FidBatchEvaluator, FidThrBatchEvaluator};
pub use batched::syncrun::{MonoExperiment, ThrExperiment};

#[cfg(feature = "mpi")]
pub mod mpi;
#[cfg(feature = "mpi")]
pub use batched::syncrun::DistExperiment;

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
        <tantale::core::experiment::MonoExperiment<_,_,_,_,_,_,_,_,_,_>
                        as tantale::core::experiment::Runable<_,_,_,_,_,_,_,_,_>>::new(
                            ($domain, $codomain),
                            $objective,
                            $optimizer,
                            $stop,
                            ($rec, $check),
                        )
    };
    (Mono, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::MonoExperiment<_,_,_,_,_,_,_,_,_,_>
                        as tantale::core::experiment::Runable<_,_,_,_,_,_,_,_,_>>::new(
                            ($domain, $codomain),
                            $objective,
                            $optimizer,
                            $stop,
                            ($rec, $check),
                        )
    };
    (Threaded, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::ThrExperiment<_,_,_,_,_,_,_,_,_,_>
                        as tantale::core::experiment::Runable<_,_,_,_,_,_,_,_,_>>::new(
                            ($domain, $codomain),
                            $objective,
                            $optimizer,
                            $stop,
                            ($rec, $check),
                        )
    };
    (Distributed, $proc:expr, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::DistExperiment<_,_,_,_,_,_,_,_,_,_,_>
                        as tantale::core::experiment::DistRunable<_,_,_,_,_,_,_,_,_>>::new(
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
        tantale::core::experiment::MonoExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_>::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Mono, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::MonoExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_>::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Threaded, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::ThrExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_>::load(($domain,$codomain), $objective, ($rec, $check))
    };
}

#[macro_export]
#[cfg(feature = "mpi")]
macro_rules! load {
    (($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::MonoExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_> as
                            tantale::core::experiment::Runable<_,_,_,_,_,_,_,_,_>
                            >::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Mono, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::MonoExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_> as
                            tantale::core::experiment::Runable<_,_,_,_,_,_,_,_,_>
                            >::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Threaded, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::ThrExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_> as
                            tantale::core::experiment::Runable<_,_,_,_,_,_,_,_,_>
                            >::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Distributed, $proc:expr, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        <tantale::core::experiment::DistExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_,_> as
                            tantale::core::experiment::DistRunable<_,_,_,_,_,_,_,_,_>
                            >::load($proc, ($domain,$codomain), $objective, ($rec, $check))
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
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SolId, Out, Scp, Op>,
    Check: Checkpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
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
pub enum MasterWorker<'a, DRun, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    DRun: DistRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
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
    Master(DRun),
    Worker(DRun::WType),
}

#[cfg(feature = "mpi")]
impl<'a, DRun, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
    MasterWorker<'a, DRun, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    DRun: DistRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
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
/// [`DistRunable`] describes a MPI-distributed [`Runable`], defined by a [`MasterWorker`] parallelization.
pub trait DistRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    Self: Sized,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    St: Stop,
    Rec: DistRecorder<PSol, SolId, Out, Scp, Op>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
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
}

//------------------------//
//-------EVALUATOR--------//
//------------------------//

pub type OutEvaluate<SolId, SInfo, Info, Scp, PSol, Cod, Out> = (
    Batch<SolId, SInfo, Info, CompShape<Scp, PSol, SolId, SInfo, Cod, Out>>,
    OutBatch<SolId, Info, Out>,
);

pub trait Evaluate
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}

/// [`SingleEvaluate`] is an [`Evaluate`] describing how to evaluate the output of a sequential [`Optimizer`]
/// generating a **single** [`Partial`] at each step.
pub trait SingleEvaluate<PSol, SolId, Op, Scp, Out, St, Fn>: Evaluate
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: SequentialOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, Fn>,
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
    ) -> OutEvaluate<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>;
    fn update(&mut self, batch: Batch<SolId, Op::SInfo, Op::Info, Scp::SolShape>);
}

/// [`MonoEvaluate`] is an [`Evaluate`] describing how to evaluate the output of a sequential [`Optimizer`]
/// generating a **batch** of [`Partial`] at each step.
pub trait MonoEvaluate<PSol, SolId, Op, Scp, Out, St, Fn>: Evaluate
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, Fn>,
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
    ) -> OutEvaluate<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>;
    fn update(&mut self, batch: Batch<SolId, Op::SInfo, Op::Info, Scp::SolShape>);
}

/// [`ThrEvaluate`] is an [`Evaluate`] describing how to evaluate, with multi-threading, the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait ThrEvaluate<PSol, SolId, Op, Scp, Out, St, Fn>: Evaluate
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, Fn>,
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
    ) -> OutEvaluate<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>;
    fn update(&mut self, batch: Batch<SolId, Op::SInfo, Op::Info, Scp::SolShape>);
}

#[cfg(feature = "mpi")]
/// [`DistEvaluate`] is an [`Evaluate`] describing how to distribute, with MPI, the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait DistEvaluate<PSol, SolId, Op, Scp, Out, St, Fn, M>: Evaluate
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, Fn>,
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
    ) -> OutEvaluate<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>;
    fn update(&mut self, batch: Batch<SolId, Op::SInfo, Op::Info, Scp::SolShape>);
}

// pub use mpi::synchronous::mpifidseqrun::{FidExperiment,FidEvaluator};
// pub use mpi::synchronous::mpifidthrrun::{FidThrExperiment,FidThrEvaluator};
