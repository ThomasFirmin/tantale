use crate::{
    OptInfo, Searchspace,
    domain::Domain,
    objective::{Codomain, FuncWrapper, Outcome},
    optimizer::Optimizer,
    saver::Saver,
    solution::{Id, SolInfo, BatchType},
    stop::Stop
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use crate::experiment::mpi::tools::MPIProcess;

pub type EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo, Info,BType> = (
    <BType as BatchType<SolId,Obj,Opt,SInfo,Info>>::Outc<Out>,
    <BType as BatchType<SolId,Obj,Opt,SInfo,Info>>::Comp<Cod,Out>,
);

pub trait Runable<SolId, Scp, Op, St, Sv, Obj, Opt, Out, Cod, Eval, FnWrap>
where
    SolId: Id,
    Scp: Searchspace<SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Eval: Evaluate,
    Sv: Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    FnWrap: FuncWrapper,
{
    fn new(searchspace: Scp,objective: FnWrap,optimizer: Op,stop: St, saver: Sv) -> Self;
    fn run(self);
    fn load(searchspace: Scp, objective: FnWrap, saver: Sv) -> Self;
}

#[cfg(feature = "mpi")]
pub trait DistRunable<SolId, Scp, Op, St, Sv, Obj, Opt, Out, Cod, Eval, FnWrap>
where
    SolId: Id,
    Scp: Searchspace<SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Eval: Evaluate,
    Sv: Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    FnWrap: FuncWrapper,
{
    fn new(proc: &MPIProcess,searchspace: Scp,objective: FnWrap,optimizer: Op,stop: St,saver: Sv) -> Self;
    fn run(self, proc: &MPIProcess);
    fn load(proc: &MPIProcess, searchspace: Scp, objective: FnWrap, saver: Sv) -> Self;
}

pub trait Evaluate
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}

/// [`SingleEvaluate`] is an [`Evaluate`] describing how to evaluate the output of a sequential [`Optimizer`]
/// generating a single [`Partial`] at each step.
pub trait SingleEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnWrap,BType>: Evaluate
where
    St: Stop,
    FnWrap: FuncWrapper,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    SolId: Id,
    FnWrap: FuncWrapper,
    BType:BatchType<SolId,Obj,Opt,SInfo,Info>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo,Info,BType>;
    fn update(&mut self,batch: BType);
}

/// [`MonoEvaluate`] is an [`Evaluate`] describing how to evaluate the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait MonoEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnWrap, BType>: Evaluate
where
    St: Stop,
    FnWrap: FuncWrapper,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    SolId: Id,
    FnWrap: FuncWrapper,
    BType: BatchType<SolId,Obj,Opt,SInfo,Info>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo,Info,BType>;
    fn update(&mut self,batch: BType);
}

/// [`ThrEvaluate`] is an [`Evaluate`] describing how to evaluate, with multi-threading, the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait ThrEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnWrap,BType>: Evaluate
where
    Self: Serialize + for<'de> Deserialize<'de>,
    St: Stop,
    FnWrap: FuncWrapper,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    SolId: Id,
    FnWrap: FuncWrapper,
    BType: BatchType<SolId,Obj,Opt,SInfo,Info>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo,Info,BType>;
    fn update(&mut self,batch: BType);
}

#[cfg(feature = "mpi")]
/// [`DistEvaluate`] is an [`Evaluate`] describing how to distribute, with MPI, the output of a sequential [`Optimizer`]
/// generating a batch of [`Partial`] at each step.
pub trait DistEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnWrap, BType>: Evaluate
where
    Self: Serialize + for<'de> Deserialize<'de>,
    St: Stop,
    FnWrap: FuncWrapper,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    SolId: Id,
    FnWrap: FuncWrapper,
    BType: BatchType<SolId,Obj,Opt,SInfo,Info>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        proc: &MPIProcess,
        ob: Arc<FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo,Info,BType>;
    fn update(&mut self,batch: BType);
}

// SYNCHRONOUS
pub mod synchronous;
pub use synchronous::evaluator::{MonoEvaluator, ThrEvaluator};
pub use synchronous::fidevaluator::{FidEvaluator, FidThrEvaluator};
pub use synchronous::fidsyncrun::FidExperiment;
pub use synchronous::syncrun::SyncExperiment;

// Utils
pub mod utils;

pub enum RunMode {
    Single,
    Monothreaded,
    Threaded,
    Distributed
}

#[cfg(feature = "mpi")]
pub mod mpi;
// pub use mpi::synchronous::mpifidseqrun::{FidExperiment,FidEvaluator};
// pub use mpi::synchronous::mpifidthrrun::{FidThrExperiment,FidThrEvaluator};
