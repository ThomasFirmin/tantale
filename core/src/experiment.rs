use crate::{
    domain::Domain,
    objective::Outcome,
    optimizer::Optimizer,
    saver::Saver,
    solution::{BatchType, Id},
    stop::Stop,
    Searchspace,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use crate::experiment::mpi::tools::MPIProcess;

// SYNCHRONOUS
pub mod synchronous;
pub use synchronous::evaluator::{BatchEvaluator, ThrBatchEvaluator};
pub use synchronous::fidevaluator::{FidEvaluator, FidThrEvaluator};
pub use synchronous::syncrun::SyncExperiment;

// Utils
pub mod utils;

#[cfg(feature = "mpi")]
pub mod mpi;

#[macro_export]
#[cfg(not(feature = "mpi"))]
macro_rules! experiment {
    (Mono, $optname: ident, $searchspace: expr ,$objective: expr, $optimizer: expr, $stop: expr, $saver: expr) => {
        $optname::get_mono($searchspace, $objective, $optimizer, $stop, $saver)
    };
    (Threaded, $optname: ident, $searchspace: expr ,$objective: expr, $optimizer: expr,$stop: expr, $saver: expr) => {
        $optname::get_threaded($searchspace, $objective, $optimizer, $stop, $saver)
    };
}

#[macro_export]
#[cfg(feature = "mpi")]
macro_rules! experiment {
    (Mono, $optname: ident | $searchspace: expr ,$objective: expr, $optimizer: expr, $stop: expr, $saver: expr) => {
        <$optname as tantale::core::optimizer::MonoOptimizer<_, _, _, _, _>>::get_mono(
            $searchspace,
            $objective,
            $optimizer,
            $stop,
            $saver,
        )
    };
    (Threaded, $optname: ident | $searchspace: expr ,$objective: expr, $optimizer: expr, $stop: expr, $saver: expr) => {
        <$optname as tantale::core::optimizer::ThrOptimizer<_, _, _, _, _>>::get_threaded(
            $searchspace,
            $objective,
            $optimizer,
            $stop,
            $saver,
        )
    };
    (Distributed, $optname: ident | $proc: expr, $searchspace: expr ,$objective: expr, $optimizer: expr,$stop: expr, $saver: expr) => {
        <$optname as tantale::core::optimizer::DistOptimizer<_, _, _, _, _>>::get_distributed(
            $proc,
            $searchspace,
            $objective,
            $optimizer,
            $stop,
            $saver,
        )
    };
}

#[macro_export]
#[cfg(not(feature = "mpi"))]
macro_rules! load {
    ($experiment: ident, $optimizer : ident, $stop : ident | $searchspace : expr, $objective : expr , $saver : expr) => {
        $experiment::<_, _, $optimizer, $stop, _, _, _, _, _, _, _>::load(
            $searchspace,
            $objective,
            $saver,
        )
    };
    ($experiment: ident, $optimizer : ident, $stop : ident | $process : expr, $searchspace : expr, $objective : expr , $saver : expr) => {
        $experiment::<_, _, $optimizer, $stop, _, _, _, _, _, _, _>::load(
            $process,
            $searchspace,
            $objective,
            $saver,
        )
    };
}

#[macro_export]
#[cfg(feature = "mpi")]
macro_rules! load {
    (Mono, $optname: ident, $stop: ident | $searchspace: expr ,$objective: expr, $saver: expr) => {
        <$optname as tantale::core::optimizer::MonoOptimizer<_,_,_,_,_>>::load_mono::<$stop,_>($searchspace,$objective,$saver)
    };
    (Threaded, $optname: ident, $stop: ident | $searchspace: expr ,$objective: expr, $saver: expr) => {
        <$optname as tantale::core::optimizer::ThrOptimizer<_,_,_,_,_>>::load_threaded::<$stop,_>($searchspace,$objective,$saver)
    };
    (Distributed, $optname: ident, $stop: ident | $searchspace: expr ,$objective: expr, $saver: expr) => {
        <$optname as tantale::core::optimizer::DistOptimizer<_,_,_,_,_>>::load_distributed::<$stop,_>($proc,$searchspace,$objective,$saver)
    };
}

pub type ExpOut<Op, SolId, Obj, Opt, Out, Scp> = EvaluateOut<
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::BType,
    SolId,
    Obj,
    Opt,
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Cod,
    Out,
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::SInfo,
    <Op as Optimizer<SolId, Obj, Opt, Out, Scp>>::Info,
>;

pub type EvaluateOut<BType, SolId, Obj, Opt, Cod, Out, SInfo, Info> = (
    <BType as BatchType<SolId, Obj, Opt, SInfo, Info>>::Outc<Out>,
    <BType as BatchType<SolId, Obj, Opt, SInfo, Info>>::Comp<Cod, Out>,
);

pub trait Runable<Eval, SolId, Scp, Op, St, Sv, Out, Obj, Opt>
where
    Eval: Evaluate,
    SolId: Id,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    St: Stop,
    Sv: Saver<SolId, St, Obj, Opt, Out, Scp, Op, Eval>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
{
    fn new(searchspace: Scp, objective: Op::FnWrap, optimizer: Op, stop: St, saver: Sv) -> Self;
    fn run(self);
    fn load(searchspace: Scp, objective: Op::FnWrap, saver: Sv) -> Self;
}

#[cfg(feature = "mpi")]
pub trait DistRunable<Eval, SolId, Scp, Op, St, Sv, Out, Obj, Opt>
where
    Eval: Evaluate,
    SolId: Id,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    St: Stop,
    Sv: Saver<SolId, St, Obj, Opt, Out, Scp, Op, Eval>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
{
    fn new_dist(
        proc: &MPIProcess,
        searchspace: Scp,
        objective: Op::FnWrap,
        optimizer: Op,
        stop: St,
        saver: Sv,
    ) -> Self;
    fn run_dist(self, proc: &MPIProcess);
    fn load_dist(proc: &MPIProcess, searchspace: Scp, objective: Op::FnWrap, saver: Sv) -> Self;
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
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Op::FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> ExpOut<Op, SolId, Obj, Opt, Out, Scp>;
    fn update(&mut self, batch: Op::BType);
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
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Op::FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> ExpOut<Op, SolId, Obj, Opt, Out, Scp>;
    fn update(&mut self, batch: Op::BType);
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
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Op::FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> ExpOut<Op, SolId, Obj, Opt, Out, Scp>;
    fn update(&mut self, batch: Op::BType);
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
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        proc: &MPIProcess,
        ob: Arc<Op::FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> ExpOut<Op, SolId, Obj, Opt, Out, Scp>;
    fn update(&mut self, batch: Op::BType);
}

// pub use mpi::synchronous::mpifidseqrun::{FidExperiment,FidEvaluator};
// pub use mpi::synchronous::mpifidthrrun::{FidThrExperiment,FidThrEvaluator};
