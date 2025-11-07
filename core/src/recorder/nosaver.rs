use crate::{
    domain::onto::OntoDom,
    objective::Outcome,
    optimizer::{CBType, OBType, Optimizer},
    recorder::Recorder,
    searchspace::Searchspace,
    solution::Id,
};

use serde::{Deserialize, Serialize};

#[cfg(feature = "mpi")]
use crate::{experiment::mpi::tools::MPIProcess, recorder::DistRecorder};

/// [`NoSaver`] does nothing, and does not save anything.
#[derive(Default, Serialize, Deserialize)]
pub struct NoSaver {}

impl NoSaver {
    pub fn new() -> NoSaver {
        NoSaver {}
    }
}

impl<SolId, Obj, Opt, Out, Scp, Op> Recorder<SolId, Obj, Opt, Out, Scp, Op> for NoSaver
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
{
    fn init(&mut self){}
    fn after_load(&mut self){}
    fn save_partial(&self, _batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>){}
    fn save_codom(&self,_batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>){}
    fn save_info(&self, _batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>){}
    fn save_out(&self, _batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>){}
}

#[cfg(feature = "mpi")]
impl<SolId, Obj, Opt, Out, Scp, Op> DistRecorder<SolId, Obj, Opt, Out, Scp, Op> for NoSaver
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
{
    fn init_dist(&mut self, _proc:&MPIProcess){}
    fn after_load_dist(&mut self, _proc:&MPIProcess){}
    fn save_partial_dist(&self, _batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>){}
    fn save_info_dist(&self, _batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>){}
    fn save_out_dist(&self, _batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>){}
    fn save_codom_dist(&self,_batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>){}
}
