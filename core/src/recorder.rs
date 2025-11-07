use crate::{
    Optimizer,
    domain::onto::OntoDom,
    objective::Outcome,
    optimizer::opt::{CBType, OBType},
    searchspace::Searchspace,
    solution::Id
};

#[cfg(feature = "mpi")]
use crate::experiment::mpi::tools::MPIProcess;

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
    fn init_dist(&mut self, proc:&MPIProcess);
    fn after_load_dist(&mut self, proc:&MPIProcess);
    fn save_partial_dist(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>);
    fn save_info_dist(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>);
    fn save_out_dist(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>);
    fn save_codom_dist(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>);
}