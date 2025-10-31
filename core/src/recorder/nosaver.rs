use crate::{
    config::NoConfig,
    domain::onto::OntoDom,
    objective::Outcome,
    optimizer::{CBType, OBType, Optimizer},
    recorder::Recorder,
    searchspace::Searchspace,
    solution::Id,
};

#[cfg(feature = "mpi")]
use mpi::Rank;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "mpi")]
use crate::recorder::DistRecorder;

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
    type Config = NoConfig;

    fn get_config(&self)->Arc<Self::Config>{}
    fn init(&mut self){}
    fn after_load(&mut self){}
    fn save_partial(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>){}
    fn save_codom(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>){}
    fn save_info(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>){}
    fn save_out(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>){}
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
    fn get_config(&self, rank:Rank)->Arc<Self::Config>{}
    fn init(&mut self, rank:Rank){}
    fn after_load(&mut self, rank:Rank){}
    fn save_partial(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, rank:Rank){}
    fn save_info(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, rank:Rank){}
    fn save_out(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>, rank:Rank){}
    fn save_codom(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, rank:Rank){}
}
