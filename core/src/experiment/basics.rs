use crate::{
    Id, Optimizer, Outcome, Searchspace, Stop, ThrCheckpointer, checkpointer::MonoCheckpointer, domain::onto::LinkOpt, experiment::Evaluate, objective::FuncWrapper, optimizer::opt::OpSInfType, recorder::Recorder, searchspace::CompShape, solution::{SolutionShape, Uncomputed, shape::RawObj}
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::DistCheckpointer, experiment::mpi::utils::MPIProcess, recorder::DistRecorder,
    solution::HasY,
};

//--------------------//
//--- MONOTHREADED ---//
//--------------------//

/// Monothreaded experiment descriptor.
///
/// This is a container that gathers all the components required to
/// execute an optimization loop in a single thread. It does **not** run the
/// optimization by itself; execution is delegated to a [`Runable`](crate::experiment::Runable)
/// implementation and an [`Evaluate`](crate::experiment::Evaluate) evaluator.
///
/// # Type parameters
/// - `PSol`: The uncomputed solution type produced by the [`Searchspace`](crate::searchspace::Searchspace).
/// - `SolId`: The identifier type for solutions, implementing [`Id`](crate::Id).
/// - `Scp`: The [`Searchspace`](crate::searchspace::Searchspace) definition.
/// - `Op`: The [`Optimizer`](crate::optimizer::Optimizer) implementation.
/// - `St`: The stopping criterion implementing [`Stop`](crate::stop::Stop).
/// - `Rec`: A [`Recorder`](crate::recorder::Recorder) implementation (optional).
/// - `Check`: A [`Checkpointer`](crate::checkpointer::Checkpointer) implementation (optional).
/// - `Out`: The [`Outcome`](crate::objective::Outcome) type describing raw outputs.
/// - `Fn`: A wrapped objective implementing [`FuncWrapper`](crate::objective::FuncWrapper).
/// - `Eval`: The evaluation strategy implementing [`Evaluate`](crate::experiment::Evaluate).
///
/// # Notes
/// - `recorder` and `checkpointer` are optional.
/// - `evaluator` is kept `pub(crate)` because it is intended to be used internally
///   by runables, not by end users.
pub struct MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SolId, Out, Scp, Op>,
    Check: MonoCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluate,
{
    pub searchspace: Scp,
    pub codomain: Op::Cod,
    pub objective: Fn,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    pub(crate) evaluator: Option<Eval>,
}

//---------------------//
//--- MULTITHREADED ---//
//---------------------//

/// Multithreaded experiment descriptor.
///
/// This container mirrors [`MonoExperiment`] but is designed for parallel
/// execution contexts. The evaluator will typically distribute the objective
/// evaluations across multiple threads.
///
/// # See also
/// - [`MonoExperiment`] for the single-threaded equivalent.
/// - [`Runable`](crate::experiment::Runable) for the execution loop.
pub struct ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SolId, Out, Scp, Op>,
    Check: ThrCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluate,
{
    pub searchspace: Scp,
    pub codomain: Op::Cod,
    pub objective: Fn,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    pub(crate) evaluator: Option<Eval>,
}

//-------------------//
//--- DISTRIBUTED ---//
//-------------------//

#[cfg(feature = "mpi")]
/// Distributed experiment descriptor for MPI backends.
///
/// This structure is enabled with the `mpi` feature and stores an [`MPIProcess`]
/// handle along with distributed variants of the recorder and checkpointer.
/// It is intended for experiments where evaluations are distributed across
/// multiple distributed processes.
///
/// # See also
/// - [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess)
/// - [`DistRecorder`](crate::recorder::DistRecorder)
/// - [`DistCheckpointer`](crate::checkpointer::DistCheckpointer)
pub struct MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    St: Stop,
    Rec: DistRecorder<PSol, SolId, Out, Scp, Op>,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluate,
{
    pub proc: &'a MPIProcess,
    pub searchspace: Scp,
    pub codomain: Op::Cod,
    pub objective: Fn,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    pub(crate) evaluator: Option<Eval>,
}
