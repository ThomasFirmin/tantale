use std::sync::Arc;

use crate::{
    Optimizer, Partial,
    domain::onto::LinkOpt,
    objective::Outcome,
    optimizer::opt::{OptCompBatch, OptCompPair},
    searchspace::{Searchspace},
    solution::{Id, IntoComputed, OutBatch}
};

#[cfg(feature = "mpi")]
use crate::{domain::onto::LinkObj, experiment::mpi::utils::MPIProcess};

pub mod csv;
pub use csv::CSVRecorder;

pub mod nosaver;
pub use nosaver::NoSaver;

pub trait Recorder<SolId, Out, Scp, Op>
where
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
{
    fn init(&mut self, scp: &Scp, cod: &Op::Cod);
    fn after_load(&mut self, scp: &Scp, cod: &Op::Cod);
    fn save_pair(&self,computed: &OptCompPair<Op,Scp,SolId,Out>,outputed: &(SolId,Out),scp: &Scp,cod: &Op::Cod, info:Arc<Op::Info>);
    fn save_batch(&self,computed: &OptCompBatch<Op,Scp,SolId,Out>,outputed: &OutBatch<SolId,Op::Info,Out>,scp: &Scp,cod: &Op::Cod);
    fn save_pair_partial(&self, pair: &OptCompPair<Op,Scp,SolId,Out>, scp: &Scp, cod: &Op::Cod);
    fn save_pair_codom(&self, pair: &OptCompPair<Op,Scp,SolId,Out>, scp: &Scp, cod: &Op::Cod);
    fn save_pair_info(&self, pair: &OptCompPair<Op,Scp,SolId,Out>, info:Arc<Op::Info>, scp: &Scp, cod: &Op::Cod);
    fn save_pair_out(&self, pair: &(SolId,Out), scp: &Scp, cod: &Op::Cod);
    fn save_batch_partial(&self, batch: &OptCompBatch<Op,Scp,SolId,Out>, scp: &Scp, cod: &Op::Cod);
    fn save_batch_codom(&self, batch: &OptCompBatch<Op,Scp,SolId,Out>, scp: &Scp, cod: &Op::Cod);
    fn save_batch_info(&self, batch: &OptCompBatch<Op,Scp,SolId,Out>, scp: &Scp, cod: &Op::Cod);
    fn save_batch_out(&self, batch: &OutBatch<SolId,Op::Info,Out>, scp: &Scp, cod: &Op::Cod);
}

#[cfg(feature = "mpi")]
pub trait DistRecorder<SolId, Out, Scp, Op>:Recorder<SolId, Out, Scp, Op>
where
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
{
    fn init_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);
    fn after_load_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);
    fn save_pair_dist(&self,computed: &OptCompPair<Op,Scp,SolId,Out>,outputed: &(SolId,Out),scp: &Scp,cod: &Op::Cod, info:Arc<Op::Info>);
    fn save_batch_dist(&self,computed: &OptCompBatch<Op,Scp,SolId,Out>,outputed: &OutBatch<SolId,Op::Info,Out>,scp: &Scp,cod: &Op::Cod);
    fn save_pair_partial_dist(&self, pair: &OptCompPair<Op,Scp,SolId,Out>, scp: &Scp, cod: &Op::Cod);
    fn save_pair_codom_dist(&self, pair: &OptCompPair<Op,Scp,SolId,Out>, scp: &Scp, cod: &Op::Cod);
    fn save_pair_info_dist(&self, pair: &OptCompPair<Op,Scp,SolId,Out>, info:Arc<Op::Info>, scp: &Scp, cod: &Op::Cod);
    fn save_pair_out_dist(&self, pair: &(SolId,Out), scp: &Scp, cod: &Op::Cod);
    fn save_batch_partial_dist(&self, batch: &OptCompBatch<Op,Scp,SolId,Out>, scp: &Scp, cod: &Op::Cod);
    fn save_batch_codom_dist(&self, batch: &OptCompBatch<Op,Scp,SolId,Out>, scp: &Scp, cod: &Op::Cod);
    fn save_batch_info_dist(&self, batch: &OptCompBatch<Op,Scp,SolId,Out>, scp: &Scp, cod: &Op::Cod);
    fn save_batch_out_dist(&self, batch: &OutBatch<SolId,Op::Info,Out>, scp: &Scp, cod: &Op::Cod);
}
