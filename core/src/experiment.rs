use crate::{
    Codomain, OptInfo, Partial, Searchspace, SolInfo, checkpointer::Checkpointer, domain::onto::OntoDom, objective::{FuncWrapper, Outcome}, optimizer::Optimizer, recorder::Recorder, solution::{BatchType, Id}, stop::Stop
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::DistCheckpointer,
    experiment::mpi::{tools::MPIProcess, worker::Worker},
    recorder::DistRecorder,
};

// SYNCHRONOUS
pub mod synchronous;
pub use synchronous::evaluator::{BatchEvaluator, ThrBatchEvaluator};
pub use synchronous::fidevaluator::{FidBatchEvaluator, FidThrBatchEvaluator};
pub use synchronous::syncrun::{MonoExperiment,ThrExperiment};

#[cfg(feature = "mpi")]
pub mod mpi;
#[cfg(feature = "mpi")]
pub use synchronous::syncrun::DistExperiment;

#[macro_export]
#[cfg(not(feature = "mpi"))]
macro_rules! experiment {
    (($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::MonoExperiment::new(($domain,$codomain), $objective, $optimizer, $stop, ($rec, $check))
    };
    (Mono, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::MonoExperiment::new(($domain,$codomain), $objective, $optimizer, $stop, ($rec, $check))
    };
    (Threaded, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::ThrExperiment::new(($domain,$codomain), $objective, $optimizer, $stop, ($rec, $check))
    };
}

#[macro_export]
#[cfg(feature = "mpi")]
macro_rules! experiment {
    (($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::MonoExperiment::new(($domain,$codomain), $objective, $optimizer, $stop, ($rec, $check))
    };
    (Mono, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::MonoExperiment::new(($domain,$codomain), $objective, $optimizer, $stop, ($rec, $check))
    };
    (Threaded, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::ThrExperiment::new(($domain,$codomain), $objective, $optimizer, $stop, ($rec, $check))
    };
    (Distributed, $proc:expr, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer: expr, $stop: expr, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::DistExperiment::new($proc, ($domain,$codomain), $objective, $optimizer, $stop, ($rec, $check))
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
        tantale::core::experiment::MonoExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_,_>::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Mono, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::MonoExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_,_>::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Threaded, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::ThrExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_,_>::load(($domain,$codomain), $objective, ($rec, $check))
    };
    (Distributed, ($domain: expr, $codomain: expr) ,$objective: expr, $optimizer:ident, $stop:ident, ($rec: expr, $check: expr)) => {
        tantale::core::experiment::DistExperiment::<_,_,_,$optimizer,$stop,_,_,_,_,_,_>::load(($domain,$codomain), $objective, ($rec, $check))
    };
}

pub type EvaluateOut<BType, SolId, Obj, Opt, Cod, Out, SInfo,  PSol, Info> = (
    <BType as BatchType<SolId, Obj, Opt, SInfo, PSol, Info>>::Outc<Out>,
    <BType as BatchType<SolId, Obj, Opt, SInfo, PSol, Info>>::Comp<Cod, Out>,
);

/// [`Runable`] describes the optimization loop.
/// The [`Optimizer`] defines a single iteration, and a [`Runable`] loops over an [`Optimizer`] step.
/// It wraps-up, the [`Optimizer`], the stopping criterion [`Stop`], the [`Objective`], the [`Recorder`] and [`Checkpointer`].
pub trait Runable<SolId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Fn>
where
    SolId: Id,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn>,
    St: Stop,
    Rec: Recorder<SolId, Obj, Opt, Out, Scp, Op, Fn, Op::BType>,
    Check: Checkpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Fn: FuncWrapper,
{
    fn new(
        space: (Scp,Op::Cod),
        objective: Fn,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>,Option<Check>)
    ) -> Self;
    fn run(self);
    fn load(space: (Scp,Op::Cod),objective: Fn,saver: (Option<Rec>,Check)) -> Self;
}

#[cfg(feature = "mpi")]
pub enum MasterWorker<'a, DRun, SolId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Fn>
where
    DRun: DistRunable<'a, SolId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Fn>,
    SolId: Id,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn>,
    St: Stop,
    Rec: DistRecorder<SolId, Obj, Opt, Out, Scp, Op, Fn, Op::BType>,
    Check: DistCheckpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Fn: FuncWrapper,
{
    Master(DRun),
    Worker(DRun::WType),
}

#[cfg(feature = "mpi")]
impl<'a, DRun, SolId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Fn>
    MasterWorker<'a, DRun, SolId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Fn>
where
    DRun: DistRunable<'a, SolId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Fn>,
    SolId: Id,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn>,
    St: Stop,
    Rec: DistRecorder<SolId, Obj, Opt, Out, Scp, Op, Fn, Op::BType>,
    Check: DistCheckpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Fn: FuncWrapper,
{
    pub fn run(self) {
        match self {
            MasterWorker::Master(exp) => exp.run_dist(),
            MasterWorker::Worker(worker) => worker.run(),
        }
    }
}

#[cfg(feature = "mpi")]
/// [`DistRunable`] describes a MPI-distributed [`Runable`], defined by a [`MasterWorker`] parallelization.
pub trait DistRunable<'a, SolId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Fn>
where
    Self: 'a + Sized,
    SolId: Id,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn>,
    St: Stop,
    Rec: DistRecorder<SolId, Obj, Opt, Out, Scp, Op, Fn, Op::BType>,
    Check: DistCheckpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Fn: FuncWrapper,
{
    type WType: Worker<SolId, Obj>;
    fn new_dist(
        proc: &'a MPIProcess,
        space: (Scp,Op::Cod),
        objective: Fn,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>,Option<Check>)
    ) -> MasterWorker<'a, Self, SolId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Fn>;
    fn run_dist(self);
    fn load_dist(proc: &'a MPIProcess,space: (Scp,Op::Cod),objective: Fn,saver: (Option<Rec>, Check)) -> MasterWorker<'a, Self, SolId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Fn>;
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
pub trait SingleEvaluate<SolId, Obj, Opt, SInfo, Info, PSol, St, Cod, Out, Scp, Fn, BType>: Evaluate
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SInfo: SolInfo,
    Info: OptInfo,
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    St: Stop,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    Fn: FuncWrapper,
    BType: BatchType<SolId,Obj,Opt,SInfo, PSol, Info>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Fn>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<BType, SolId, Obj, Opt, Cod, Out, SInfo, PSol, Info>;
    fn update(&mut self, batch: BType);
}

/// [`MonoEvaluate`] is an [`Evaluate`] describing how to evaluate the output of a sequential [`Optimizer`]
/// generating a **batch** of [`Partial`] at each step.
pub trait MonoEvaluate<SolId, Obj, Opt, SInfo, Info, PSol, St, Cod, Out, Scp, Fn, BType>: Evaluate
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SInfo: SolInfo,
    Info: OptInfo,
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    St: Stop,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    Fn: FuncWrapper,
    BType: BatchType<SolId,Obj,Opt,SInfo, PSol, Info>,
{
    fn init(&mut self);
    fn evaluate(&mut self, ob: &Fn, cod:&Cod, stop: &mut St)
        -> EvaluateOut<BType, SolId, Obj, Opt, Cod, Out, SInfo, PSol, Info>;
    fn update(&mut self, batch: BType);
}

/// [`ThrEvaluate`] is an [`Evaluate`] describing how to evaluate, with multi-threading, the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait ThrEvaluate<SolId, Obj, Opt, SInfo, Info, PSol, St, Cod, Out, Scp, Fn, BType>: Evaluate
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SInfo: SolInfo,
    Info: OptInfo,
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    St: Stop,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    Fn: FuncWrapper,
    BType: BatchType<SolId,Obj,Opt,SInfo, PSol, Info>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Fn>,
        cod: Arc<Cod>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<BType, SolId, Obj, Opt, Cod, Out, SInfo, PSol, Info>;
    fn update(&mut self, batch: BType);
}

#[cfg(feature = "mpi")]
/// [`DistEvaluate`] is an [`Evaluate`] describing how to distribute, with MPI, the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait DistEvaluate<SolId, Obj, Opt, SInfo, Info, PSol, St, Cod, Out, Scp, Fn, BType>: Evaluate
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SInfo: SolInfo,
    Info: OptInfo,
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    St: Stop,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    Fn: FuncWrapper,
    BType: BatchType<SolId,Obj,Opt,SInfo, PSol, Info>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        proc: &MPIProcess,
        ob: &Fn,
        cod: &Cod,
        stop: &mut St,
    ) -> EvaluateOut<BType, SolId, Obj, Opt, Cod, Out, SInfo, PSol, Info>;
    fn update(&mut self, batch: BType);
}

// pub use mpi::synchronous::mpifidseqrun::{FidExperiment,FidEvaluator};
// pub use mpi::synchronous::mpifidthrrun::{FidThrExperiment,FidThrEvaluator};
