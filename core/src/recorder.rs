use crate::{
    domain::onto::OntoDom,
    objective::{FuncWrapper, Outcome},
    searchspace::Searchspace,
    solution::{BatchType, Id},
    Optimizer,
};

#[cfg(feature = "mpi")]
use crate::experiment::mpi::utils::MPIProcess;

pub mod csv;
pub use csv::CSVRecorder;

pub mod nosaver;
pub use nosaver::NoSaver;

pub trait Recorder<SolId, Obj, Opt, Out, Scp, Op, Fn, BType>
where
    Self: Sized,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn, BType = BType>,
    Fn: FuncWrapper,
    BType: BatchType<SolId, Obj, Opt, Op::SInfo, Op::Sol, Op::Info>,
{
    fn init(&mut self, scp: &Scp, cod: &Op::Cod);
    fn after_load(&mut self, scp: &Scp, cod: &Op::Cod);
    fn save(
        &self,
        cbatch: &BType::Comp<Op::Cod, Out>,
        obatch: &BType::Outc<Out>,
        scp: &Scp,
        cod: &Op::Cod,
    );
    fn save_partial(&self, batch: &BType::Comp<Op::Cod, Out>, scp: &Scp, cod: &Op::Cod);
    fn save_codom(&self, batch: &BType::Comp<Op::Cod, Out>, scp: &Scp, cod: &Op::Cod);
    fn save_info(&self, batch: &BType::Comp<Op::Cod, Out>, scp: &Scp, cod: &Op::Cod);
    fn save_out(&self, batch: &BType::Outc<Out>, scp: &Scp, cod: &Op::Cod);
}

#[cfg(feature = "mpi")]
pub trait DistRecorder<SolId, Obj, Opt, Out, Scp, Op, Fn, BType>:
    Recorder<SolId, Obj, Opt, Out, Scp, Op, Fn, BType>
where
    Self: Sized,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn, BType = BType>,
    Fn: FuncWrapper,
    BType: BatchType<SolId, Obj, Opt, Op::SInfo, Op::Sol, Op::Info>,
{
    fn init_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);
    fn after_load_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);
    fn save_dist(
        &self,
        cbatch: &BType::Comp<Op::Cod, Out>,
        obatch: &BType::Outc<Out>,
        scp: &Scp,
        cod: &Op::Cod,
    );
    fn save_partial_dist(&self, batch: &BType::Comp<Op::Cod, Out>, scp: &Scp, cod: &Op::Cod);
    fn save_codom_dist(&self, batch: &BType::Comp<Op::Cod, Out>, scp: &Scp, cod: &Op::Cod);
    fn save_info_dist(&self, batch: &BType::Comp<Op::Cod, Out>, scp: &Scp, cod: &Op::Cod);
    fn save_out_dist(&self, batch: &BType::Outc<Out>, scp: &Scp, cod: &Op::Cod);
}
