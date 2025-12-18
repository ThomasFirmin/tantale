use std::sync::Arc;

use crate::{
    Optimizer, domain::onto::LinkOpt, objective::Outcome, searchspace::{CompShape, Searchspace}, solution::{Batch, Id, OutBatch, SolutionShape}
};

#[cfg(feature = "mpi")]
use crate::{experiment::mpi::utils::MPIProcess, solution::HasY};

pub mod csv;
pub use csv::CSVRecorder;

pub mod nosaver;
pub use nosaver::NoSaver;

pub trait Recorder<SolId, Out, Scp, Op>
where
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasY<Op::Cod,Out>,
{
    fn init(&mut self, scp: &Scp, cod: &Op::Cod);
    fn after_load(&mut self, scp: &Scp, cod: &Op::Cod);
    fn save_pair(&self,computed: &CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>,outputed: &(SolId,Out),scp: &Scp,cod: &Op::Cod, info:Arc<Op::Info>);
    fn save_batch(&self,computed: &Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,outputed: &OutBatch<SolId,Op::Info,Out>,scp: &Scp,cod: &Op::Cod);
}

#[cfg(feature = "mpi")]
pub trait DistRecorder<SolId, Out, Scp, Op>:Recorder<SolId, Out, Scp, Op>
where
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasY<Op::Cod,Out>,
{
    fn init_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);
    fn after_load_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);
    fn save_pair_dist(&self,computed: &CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>,outputed: &(SolId,Out),scp: &Scp,cod: &Op::Cod, info:Arc<Op::Info>);
    fn save_batch_dist(&self,computed: &Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,outputed: &OutBatch<SolId,Op::Info,Out>,scp: &Scp,cod: &Op::Cod);
}
