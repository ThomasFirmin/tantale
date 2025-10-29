use crate::{
    Codomain, Optimizer, Partial, SolInfo,
    domain::onto::OntoDom,
    objective::Outcome,
    optimizer::opt::{CBType, OBType},
    searchspace::Searchspace,
    solution::Id,
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
    type Config: SaverConfig;
    fn init(&mut self, sp: &Scp, cod: &Op::Cod);
    fn after_load(&mut self, sp: &Scp, cod: &Op::Cod);
    fn save_partial(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>);
    fn save_info(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>);
    fn save_out(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>);
    fn save_codom(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>,sp: Arc<Scp>,cod: Arc<Op::Cod>);
}

#[cfg(feature = "mpi")]
pub trait DistRecorder<SolId, Obj, Opt, Out, Scp, Op>
where
    Self: Sized,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
{
    type Config: DistSaverConfig;
    fn init(&mut self, sp: &Scp, cod: &Op::Cod, rank:Rank);
    fn after_load(&mut self, sp: &Scp, cod: &Op::Cod, rank:Rank);
    fn save_partial(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>, rank:Rank);
    fn save_info(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>, rank:Rank);
    fn save_out(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>, rank:Rank);
    fn save_codom(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>,sp: Arc<Scp>,cod: Arc<Op::Cod>, rank:Rank);
}