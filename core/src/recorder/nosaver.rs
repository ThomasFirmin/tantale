use crate::{
    domain::onto::LinkOpt,
    objective::Outcome,
    optimizer::Optimizer,
    recorder::Recorder,
    searchspace::{CompShape, Searchspace},
    solution::{HasY, Id, OutBatch, SolutionShape, Uncomputed},
};

use serde::{Deserialize, Serialize};

#[cfg(feature = "mpi")]
use crate::{experiment::mpi::utils::MPIProcess, recorder::DistRecorder};

/// [`NoSaver`] does nothing, and does not save anything.
#[derive(Default, Serialize, Deserialize)]
pub struct NoSaver {}

impl NoSaver {
    pub fn new() -> Option<NoSaver> {
        Some(NoSaver {})
    }
}

impl<PSol, SolId, Out, Scp, Op> Recorder<PSol, SolId, Out, Scp, Op> for NoSaver
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
{
    fn init_seq<FnWrap:crate::FuncWrapper<crate::RawObj<Scp::SolShape, SolId, Op::SInfo>>>(&mut self, _scp: &Scp, _cod: &Op::Cod)
    where
        Op: crate::SequentialOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap> {
        
    }

    fn after_load_seq<FnWrap:crate::FuncWrapper<crate::RawObj<Scp::SolShape, SolId, Op::SInfo>>>(&mut self, _scp: &Scp, _cod: &Op::Cod)
    where
        Op: crate::SequentialOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap> {
        
    }

    fn save_pair(
        &self,
        _computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        _outputed: &(SolId, Out),
        _scp: &Scp,
        _cod: &Op::Cod,
    ) {
        
    }

    fn init_batch<FnWrap:crate::FuncWrapper<crate::RawObj<Scp::SolShape, SolId, Op::SInfo>>>(&mut self, _scp: &Scp, _cod: &Op::Cod)
    where
        Op: crate::BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap> {
        
    }

    fn after_load_batch<FnWrap:crate::FuncWrapper<crate::RawObj<Scp::SolShape, SolId, Op::SInfo>>>(&mut self, _scp: &Scp, _cod: &Op::Cod)
    where
        Op: crate::BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap> {
        
    }

    fn save_batch<FnWrap:crate::FuncWrapper<crate::RawObj<Scp::SolShape, SolId, Op::SInfo>>>
    (
        &self,
        _computed: &crate::CompBatch<SolId, Op::SInfo, <Op>::Info, Scp, PSol, Op::Cod, Out>,
        _outputed: &OutBatch<SolId, <Op>::Info, Out>,
        _scp: &Scp,
        _cod: &Op::Cod,
    )
    where
        Op: crate::BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap>,
        <Op>::Info: Send + Sync {
        
    }
}

#[cfg(feature = "mpi")]
impl<PSol, SolId, Out, Scp, Op> DistRecorder<PSol, SolId, Out, Scp, Op> for NoSaver
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
{
    fn init_seq_dist<FnWrap:crate::FuncWrapper<crate::RawObj<Scp::SolShape, SolId, Op::SInfo>>>(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod)
    where
        Op: crate::SequentialOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap> {
        
    }

    fn after_load_seq_dist<FnWrap:crate::FuncWrapper<crate::RawObj<Scp::SolShape, SolId, Op::SInfo>>>(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod)
    where
        Op: crate::SequentialOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap> {
        
    }

    fn save_pair_dist(
        &self,
        _computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        _outputed: &(SolId, Out),
        _scp: &Scp,
        _cod: &Op::Cod,
    ) {
        
    }

    fn init_batch_dist<FnWrap:crate::FuncWrapper<crate::RawObj<Scp::SolShape, SolId, Op::SInfo>>>(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod)
    where
        Op: crate::BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap> {
        
    }

    fn after_load_batch_dist<FnWrap:crate::FuncWrapper<crate::RawObj<Scp::SolShape, SolId, Op::SInfo>>>(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod)
    where
        Op: crate::BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap> {
        
    }

    fn save_batch_dist<FnWrap:crate::FuncWrapper<crate::RawObj<Scp::SolShape, SolId, Op::SInfo>>>
    (
        &self,
        _computed: &crate::CompBatch<SolId, Op::SInfo, <Op>::Info, Scp, PSol, Op::Cod, Out>,
        _outputed: &OutBatch<SolId, <Op>::Info, Out>,
        _scp: &Scp,
        _cod: &Op::Cod,
    )
    where
        Op: crate::BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap>,
        <Op>::Info: Send + Sync {
        
    }
}
