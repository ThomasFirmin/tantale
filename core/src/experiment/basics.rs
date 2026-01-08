use crate::{Id, Optimizer, Outcome, Searchspace, Stop, checkpointer::Checkpointer, domain::onto::LinkOpt, experiment::Evaluate, objective::FuncWrapper, optimizer::opt::OpSInfType, recorder::Recorder, searchspace::CompShape, solution::{SolutionShape, Uncomputed, shape::RawObj}};

#[cfg(feature = "mpi")]
use crate::{checkpointer::DistCheckpointer, recorder::DistRecorder, solution::HasY, experiment::mpi::utils::MPIProcess};

//--------------------//
//--- MONOTHREADED ---//
//--------------------//

pub struct MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SolId, Out, Scp, Op>,
    Check: Checkpointer,
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

pub struct ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SolId, Out, Scp, Op>,
    Check: Checkpointer,
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