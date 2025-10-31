use crate::{
    Optimizer, SaverConfig,
    domain::onto::OntoDom,
    objective::Outcome,
    optimizer::opt::{CBType, OBType},
    searchspace::Searchspace,
    solution::Id
};

use std::sync::Arc;
#[cfg(feature = "mpi")]
use mpi::Rank;

pub mod csv;
pub use csv::CSVRecorder;

pub mod nosaver;
pub use nosaver::NoSaver;

pub trait Recorder<SolId, Obj, Opt, Out, Scp, Op>
where
    Self: Sized,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
{
    type Config: SaverConfig<Op::Sol,Scp,SolId,Obj,Opt,Op::SInfo,Op::Cod,Out>;
    fn get_config(&self)->Arc<Self::Config>;
    fn init(&mut self);
    fn after_load(&mut self);
    fn save_partial(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>);
    fn save_codom(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>);
    fn save_info(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>);
    fn save_out(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>);
}

#[cfg(feature = "mpi")]
pub trait DistRecorder<SolId, Obj, Opt, Out, Scp, Op>: Recorder<SolId, Obj, Opt, Out, Scp, Op>
where
    Self: Sized,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
{
    fn get_config(&self, rank:Rank)->Arc<Self::Config>;
    fn init(&mut self, rank:Rank);
    fn after_load(&mut self, rank:Rank);
    fn save_partial(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, rank:Rank);
    fn save_info(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, rank:Rank);
    fn save_out(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>, rank:Rank);
    fn save_codom(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, rank:Rank);
}