use std::sync::Arc;

use crate::{
    domain::onto::LinkOpt,
    objective::Outcome,
    optimizer::opt::CompBatch,
    searchspace::{CompShape, Searchspace},
    solution::{HasY, Id, OutBatch, SolutionShape, Uncomputed},
    Optimizer,
};

#[cfg(feature = "mpi")]
use crate::experiment::mpi::utils::MPIProcess;

pub mod csv;
pub use csv::CSVRecorder;

pub mod nosaver;
pub use nosaver::NoSaver;

pub trait Recorder<PSol, SolId, Out, Scp, Op>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
{
    fn init(&mut self, scp: &Scp, cod: &Op::Cod);
    fn after_load(&mut self, scp: &Scp, cod: &Op::Cod);
    fn save_pair(
        &self,
        computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        outputed: &(SolId, Out),
        scp: &Scp,
        cod: &Op::Cod,
        info: Option<Arc<Op::Info>>,
    );
    fn save_batch(
        &self,
        computed: &CompBatch<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
        outputed: &OutBatch<SolId, Op::Info, Out>,
        scp: &Scp,
        cod: &Op::Cod,
    );
}

#[cfg(feature = "mpi")]
pub trait DistRecorder<PSol, SolId, Out, Scp, Op>: Recorder<PSol, SolId, Out, Scp, Op>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
{
    fn init_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);
    fn after_load_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);
    fn save_pair_dist(
        &self,
        computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        outputed: &(SolId, Out),
        scp: &Scp,
        cod: &Op::Cod,
        info: Option<Arc<Op::Info>>,
    );
    fn save_batch_dist(
        &self,
        computed: &CompBatch<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
        outputed: &OutBatch<SolId, Op::Info, Out>,
        scp: &Scp,
        cod: &Op::Cod,
    );
}
