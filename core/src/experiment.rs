use crate::{
    LinkedOutcome, OptInfo, Searchspace, domain::Domain, experiment::mpi::tools::MPIProcess, objective::{Codomain, FuncWrapper, Outcome}, optimizer::opt::{ArcVecArc, Optimizer, SolPairs}, saver::Saver, solution::{Id, Partial, SolInfo}, stop::Stop
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub type EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo> = (
    SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
    Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
);

pub trait Runable<SolId, Scp, Op, St, Sv, Obj, Opt, Out, Cod, Eval, FnWrap>
where
    SolId: Id,
    Scp: Searchspace<SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Eval: Evaluate<St, Obj, Opt, Out, Cod, Op::Info, Op::SInfo, SolId, FnWrap>,
    Sv: Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval, FnWrap>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    FnWrap: FuncWrapper,
{
    fn run(self);
    fn load(searchspace: Scp, objective: FnWrap, saver: Sv) -> Self;
}

/// An evaluator describes how a batch of [`Partial`] should
/// be evaluated to get a batch of [`Computed`].
pub trait Evaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnWrap>
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
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo>;
    fn update(
        &mut self,
        obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    );
}

/// Describes a multi-threaded [`Evaluate`].
pub trait ThrEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnWrap>: Evaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnWrap>
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
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        proc:&MPIProcess,
        ob: Arc<FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo>;
    fn update(
        &mut self,
        obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    );
}


#[cfg(feature="mpi")]
/// Describes an MPI-distributed [`Evaluate`].
pub trait DistEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnWrap>: Evaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnWrap>
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
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        proc:&MPIProcess,
        ob: Arc<FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo>;
    fn update(
        &mut self,
        obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    );
}

// SYNCHRONOUS
pub mod synchronous;
pub use synchronous::seqrun::{Experiment,Evaluator};
pub use synchronous::thrrun::{ThrExperiment,ThrEvaluator};
pub use synchronous::fidseqrun::{FidExperiment,FidEvaluator};
pub use synchronous::fidthrrun::{FidThrExperiment,FidThrEvaluator};

#[cfg(feature="mpi")]
pub mod mpi;
#[cfg(feature="mpi")]
pub use mpi::synchronous::mpiseqrun::{MPIExperiment,MPIEvaluator};
#[cfg(feature="mpi")]
pub use mpi::synchronous::mpithrrun::{MPIThrExperiment,MPIThrEvaluator};
// pub use mpi::synchronous::mpifidseqrun::{FidExperiment,FidEvaluator};
// pub use mpi::synchronous::mpifidthrrun::{FidThrExperiment,FidThrEvaluator};