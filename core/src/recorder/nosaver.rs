use crate::{
    domain::onto::LinkOpt, objective::Outcome, optimizer::Optimizer, recorder::Recorder, searchspace::{CompShape, Searchspace}, solution::{HasY, Id, OutBatch, SolutionShape, Uncomputed,Batch}
};

use std::sync::Arc;
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

impl<PSol,SolId, Out, Scp, Op> Recorder<PSol,SolId, Out, Scp, Op> for NoSaver
where
    PSol: Uncomputed<SolId,Scp::Opt,Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<PSol,SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<PSol,SolId,Op::SInfo>,
    CompShape<Scp,PSol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasY<Op::Cod,Out>,
{
    fn init(&mut self, _scp: &Scp, _cod: &Op::Cod) {}
    fn after_load(&mut self, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_pair(&self,_computed: &CompShape<Scp,PSol,SolId,Op::SInfo,Op::Cod,Out>,_outputed: &(SolId,Out),_scp: &Scp,_cod: &Op::Cod, _info:Arc<Op::Info>) {}
    fn save_batch(&self,_computed: &Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,PSol,SolId,Op::SInfo,Op::Cod,Out>>,_outputed: &OutBatch<SolId,Op::Info,Out>,_scp: &Scp,_cod: &Op::Cod) {}
}

#[cfg(feature = "mpi")]
impl<PSol,SolId, Out, Scp, Op> DistRecorder<PSol,SolId, Out, Scp, Op> for NoSaver
where
    PSol: Uncomputed<SolId,Scp::Opt,Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<PSol,SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<PSol,SolId,Op::SInfo>,
    CompShape<Scp,PSol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasY<Op::Cod,Out>,
{
    fn init_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {}
    fn after_load_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_pair_dist(&self,_computed: &CompShape<Scp,PSol,SolId,Op::SInfo,Op::Cod,Out>,_outputed: &(SolId,Out),_scp: &Scp,_cod: &Op::Cod, _info:Arc<Op::Info>) {}
    fn save_batch_dist(&self,_computed: &Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,PSol,SolId,Op::SInfo,Op::Cod,Out>>,_outputed: &OutBatch<SolId,Op::Info,Out>,_scp: &Scp,_cod: &Op::Cod) {}
}
