#[cfg(feature = "mpi")]
use std::sync::Arc;

use crate::{
    domain::onto::LinkOpt, objective::Outcome, optimizer::Optimizer, recorder::Recorder, searchspace::{CompShape, Searchspace}, solution::{HasY, Id, OutBatch, SolutionShape}
};

use serde::{Deserialize, Serialize};

#[cfg(feature = "mpi")]
use crate::{experiment::mpi::utils::MPIProcess, recorder::DistRecorder, solution::Batch};

/// [`NoSaver`] does nothing, and does not save anything.
#[derive(Default, Serialize, Deserialize)]
pub struct NoSaver {}

impl NoSaver {
    pub fn new() -> Option<NoSaver> {
        Some(NoSaver {})
    }
}

impl<SolId, Out, Scp, Op> Recorder<SolId, Out, Scp, Op> for NoSaver
where
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasY<Op::Cod,Out>,
{
    fn init(&mut self, _scp: &Scp, _cod: &Op::Cod) {}
    fn after_load(&mut self, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_pair(&self,_computed: &CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>,_outputed: &(SolId,Out),_scp: &Scp,_cod: &Op::Cod, _info:Arc<Op::Info>) {}
    fn save_batch(&self,_computed: &Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,_outputed: &OutBatch<SolId,Op::Info,Out>,_scp: &Scp,_cod: &Op::Cod) {}
}

#[cfg(feature = "mpi")]
impl<SolId, Out, Scp, Op> DistRecorder<SolId, Out, Scp, Op> for NoSaver
where
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasY<Op::Cod,Out>,
{
    fn init_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {}
    fn after_load_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_pair_dist(&self,_computed: &CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>,_outputed: &(SolId,Out),_scp: &Scp,_cod: &Op::Cod, _info:Arc<Op::Info>) {}
    fn save_batch_dist(&self,_computed: &Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,_outputed: &OutBatch<SolId,Op::Info,Out>,_scp: &Scp,_cod: &Op::Cod) {}
}
