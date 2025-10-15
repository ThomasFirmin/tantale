use crate::{
    Searchspace,
    domain::Domain,
    objective::Outcome,
    optimizer::{Optimizer, opt::{OpCodType, OpInfType, OpSInfType}},
    saver::Saver,
    solution::{BatchType, Id},
    stop::Stop
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use crate::experiment::mpi::tools::MPIProcess;

pub type EvaluateOut<Op,SolId, Obj, Opt, Out, Scp> = (
    <<Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::BType as BatchType<SolId,Obj,Opt,OpSInfType<Op,SolId,Obj,Opt,Out,Scp>,OpInfType<Op,SolId,Obj,Opt,Out,Scp>>>::Outc<Out>,
    <<Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::BType as BatchType<SolId,Obj,Opt,OpSInfType<Op,SolId,Obj,Opt,Out,Scp>,OpInfType<Op,SolId,Obj,Opt,Out,Scp>>>::Comp<OpCodType<Op,SolId,Obj,Opt,Out,Scp>,Out>,
);

pub trait Runable<SolId, Scp, Op, St, Sv, Out, Obj, Opt>
where
    SolId: Id,
    Scp: Searchspace<Op::Sol<Obj,Opt>,Op::Sol<Opt,Obj>,SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    St: Stop,
    Sv: Saver<SolId, St, Obj, Opt, Out, Scp, Op, Self::Eval>,
    Obj: Domain,
    Opt: Domain,
    Out:Outcome,
{
    type Eval: Evaluate;
    fn new(searchspace: Scp,objective: Op::FnWrap,optimizer: Op,stop: St, saver: Sv) -> Self;
    fn run(self);
    fn load(searchspace: Scp, objective: Op::FnWrap, saver: Sv) -> Self;
}

#[cfg(feature = "mpi")]
pub trait DistRunable<SolId, Scp, Op, St, Sv, Out, Obj, Opt>
where
    SolId: Id,
    Scp: Searchspace<Op::Sol<Obj,Opt>,Op::Sol<Opt,Obj>,SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    St: Stop,
    Sv: Saver<SolId, St, Obj, Opt, Out, Scp, Op, Self::Eval>,
    Obj: Domain,
    Opt: Domain,
    Out:Outcome,
{
    type Eval: Evaluate;
    fn new(proc:&MPIProcess,searchspace: Scp,objective: Op::FnWrap,optimizer: Op,stop: St, saver: Sv) -> Self;
    fn run(self,proc:&MPIProcess);
    fn load(proc:&MPIProcess, searchspace: Scp, objective: Op::FnWrap, saver: Sv) -> Self;
}

pub trait Evaluate
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}

/// [`SingleEvaluate`] is an [`Evaluate`] describing how to evaluate the output of a sequential [`Optimizer`]
/// generating a single [`Partial`] at each step.
pub trait SingleEvaluate<Op, St, Obj, Opt, Out, SolId, Scp>: Evaluate
where
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    SolId: Id,
    Scp: Searchspace<Op::Sol<Obj,Opt>,Op::Sol<Opt,Obj>,SolId,Obj,Opt,Op::SInfo>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Op::FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<Op,SolId, Obj, Opt,Out,Scp>;
    fn update(&mut self,batch: Op::BType);
}

/// [`MonoEvaluate`] is an [`Evaluate`] describing how to evaluate the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait MonoEvaluate<Op, St, Obj, Opt, Out, SolId, Scp>: Evaluate
where
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    SolId: Id,
    Scp: Searchspace<Op::Sol<Obj,Opt>,Op::Sol<Opt,Obj>,SolId,Obj,Opt,Op::SInfo>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Op::FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<Op,SolId, Obj, Opt,Out,Scp>;
    fn update(&mut self,batch: Op::BType);
}

/// [`ThrEvaluate`] is an [`Evaluate`] describing how to evaluate, with multi-threading, the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait ThrEvaluate<Op, St, Obj, Opt, Out, SolId, Scp>: Evaluate
where
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    SolId: Id,
    Scp: Searchspace<Op::Sol<Obj,Opt>,Op::Sol<Opt,Obj>,SolId,Obj,Opt,Op::SInfo>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Op::FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<Op,SolId, Obj, Opt,Out,Scp>;
    fn update(&mut self,batch: Op::BType);
}

#[cfg(feature = "mpi")]
/// [`DistEvaluate`] is an [`Evaluate`] describing how to distribute, with MPI, the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait DistEvaluate<Op, St, Obj, Opt, Out, SolId, Scp>: Evaluate
where
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    SolId: Id,
    Scp: Searchspace<Op::Sol<Obj,Opt>,Op::Sol<Opt,Obj>,SolId,Obj,Opt,Op::SInfo>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        proc: &MPIProcess,
        ob: Arc<Op::FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<Op,SolId, Obj, Opt,Out,Scp>;
    fn update(&mut self,batch: Op::BType);
}

// SYNCHRONOUS
pub mod synchronous;
pub use synchronous::evaluator::{MonoEvaluator, ThrEvaluator};
pub use synchronous::fidevaluator::{FidEvaluator, FidThrEvaluator};
pub use synchronous::fidsyncrun::FidExperiment;
pub use synchronous::syncrun::SyncExperiment;

// Utils
pub mod utils;

#[cfg(feature = "mpi")]
pub mod mpi;
// pub use mpi::synchronous::mpifidseqrun::{FidExperiment,FidEvaluator};
// pub use mpi::synchronous::mpifidthrrun::{FidThrExperiment,FidThrEvaluator};
