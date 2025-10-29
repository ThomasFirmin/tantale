use crate::{
    GlobalParameters, Optimizer, SaverConfig,
    domain::onto::OntoDom,
    experiment::Evaluate,
    objective::Outcome,
    searchspace::Searchspace,
    solution::Id,
    stop::Stop
};
use std::sync::Arc;

#[cfg(feature = "mpi")]
use crate::DistSaverConfig;
#[cfg(feature = "mpi")]
use mpi::Rank;

pub mod checkerror;
pub use checkerror::CheckpointError;

pub trait Checkpointer<SolId, St, Obj, Opt, Out, Scp, Op, Eval>
where
    Self: Sized,
    SolId: Id,
    St: Stop,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    Eval: Evaluate,
{
    type Config: SaverConfig;
    fn init(&mut self, sp: &Scp, cod: &Op::Cod);
    fn after_load(&mut self, sp: &Scp, cod: &Op::Cod);
    fn save_state(&self, sp: Arc<Scp>, state: &Op::State, stop: &St, eval: &Eval);
    fn load(&self, sp: &Scp, cod: &Op::Cod) -> Result<(St, Op, Eval), CheckpointError>;
    fn load_stop(&self, sp: &Scp, cod: &Op::Cod) -> Result<St, CheckpointError>;
    fn load_optimizer(&self, sp: &Scp, cod: &Op::Cod) -> Result<Op, CheckpointError>;
    fn load_evaluate(&self, sp: &Scp, cod: &Op::Cod) -> Result<Eval, CheckpointError>;
    fn load_parameters(&self,sp: &Scp,cod: &Op::Cod) -> Result<GlobalParameters, CheckpointError>;
}

#[cfg(feature = "mpi")]
pub trait DistCheckpointer<SolId, St, Obj, Opt, Out, Scp, Op, Eval>
where
    Self: Sized,
    SolId: Id,
    St: Stop,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    Eval: Evaluate,
{
    type Config: DistSaverConfig;
    fn init(&mut self, sp: &Scp, cod: &Op::Cod, rank:Rank);
    fn after_load(&mut self, sp: &Scp, cod: &Op::Cod, rank:Rank);
    fn save_state(&self, sp: Arc<Scp>, state: &Op::State, stop: &St, eval: &Eval, rank:Rank);
    fn load(&self, sp: &Scp, cod: &Op::Cod, rank:Rank) -> Result<(St, Op, Eval), CheckpointError>;
    fn load_stop(&self, sp: &Scp, cod: &Op::Cod, rank:Rank) -> Result<St, CheckpointError>;
    fn load_optimizer(&self, sp: &Scp, cod: &Op::Cod, rank:Rank) -> Result<Op, CheckpointError>;
    fn load_evaluate(&self, sp: &Scp, cod: &Op::Cod, rank:Rank) -> Result<Eval, CheckpointError>;
    fn load_parameters(&self,sp: &Scp,cod: &Op::Cod, rank:Rank) -> Result<GlobalParameters, CheckpointError>;
}