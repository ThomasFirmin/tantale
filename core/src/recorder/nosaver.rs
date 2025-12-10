#[cfg(feature = "mpi")]
use std::sync::Arc;

use crate::{
    Partial, domain::onto::{LinkObj, LinkOpt}, objective::Outcome, optimizer::{Optimizer, opt::{OptCompBatch, OptCompPair}}, recorder::Recorder, searchspace::Searchspace, solution::{Id, IntoComputed, OutBatch}
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

impl<SolId, Out, Scp, Op> Recorder<SolId, Out, Scp, Op> for NoSaver
where
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
{
    fn init(&mut self, _scp: &Scp, _cod: &Op::Cod) {}
    fn after_load(&mut self, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_pair(&self,_computed: &OptCompPair<Op,Scp,SolId,Out>,_outputed: &(SolId,Out),_scp: &Scp,_cod: &Op::Cod, _info:Arc<Op::Info>) {}
    fn save_batch(&self,_computed: &OptCompBatch<Op,Scp,SolId,Out>,_outputed: &OutBatch<SolId,Op::Info,Out>,_scp: &Scp,_cod: &Op::Cod) {}
    fn save_pair_partial(&self, _pair: &OptCompPair<Op,Scp,SolId,Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_pair_codom(&self, _pair: &OptCompPair<Op,Scp,SolId,Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_pair_info(&self, _pair: &OptCompPair<Op,Scp,SolId,Out>, _info:Arc<Op::Info>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_pair_out(&self, _pair: &(SolId,Out), _scp: &Scp, _cod: &Op::Cod) {}
    fn save_batch_partial(&self, _batch: &OptCompBatch<Op,Scp,SolId,Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_batch_codom(&self, _batch: &OptCompBatch<Op,Scp,SolId,Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_batch_info(&self, _batch: &OptCompBatch<Op,Scp,SolId,Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_batch_out(&self, _batch: &OutBatch<SolId,Op::Info,Out>, _scp: &Scp, _cod: &Op::Cod) {}}

#[cfg(feature = "mpi")]
impl<SolId, Out, Scp, Op> DistRecorder<SolId, Out, Scp, Op> for NoSaver
where
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
{
    fn init_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {}
    fn after_load_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_pair_dist(&self,_computed: &OptCompPair<Op,Scp,SolId,Out>,_outputed: &(SolId,Out),_scp: &Scp,_cod: &Op::Cod, _info:Arc<Op::Info>) {}
    fn save_batch_dist(&self,_computed: &OptCompBatch<Op,Scp,SolId,Out>,_outputed: &OutBatch<SolId,Op::Info,Out>,_scp: &Scp,_cod: &Op::Cod) {}
    fn save_pair_partial_dist(&self, _pair: &OptCompPair<Op,Scp,SolId,Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_pair_codom_dist(&self, _pair: &OptCompPair<Op,Scp,SolId,Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_pair_info_dist(&self, _pair: &OptCompPair<Op,Scp,SolId,Out>, _info:Arc<Op::Info>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_pair_out_dist(&self, _pair: &(SolId,Out), _scp: &Scp, _cod: &Op::Cod) {}
    fn save_batch_partial_dist(&self, _batch: &OptCompBatch<Op,Scp,SolId,Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_batch_codom_dist(&self, _batch: &OptCompBatch<Op,Scp,SolId,Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_batch_info_dist(&self, _batch: &OptCompBatch<Op,Scp,SolId,Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_batch_out_dist(&self, _batch: &OutBatch<SolId,Op::Info,Out>, _scp: &Scp, _cod: &Op::Cod) {}}
