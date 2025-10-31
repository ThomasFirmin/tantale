use crate::{
    GlobalParameters, Optimizer, SaverConfig,
    domain::onto::OntoDom,
    experiment::Evaluate,
    objective::Outcome,
    searchspace::Searchspace,
    solution::Id,
    stop::Stop
};

#[cfg(feature = "mpi")]
use mpi::Rank;

pub mod checkerror;
pub use checkerror::CheckpointError;

pub mod messagepack;
pub use messagepack::MessagePack;

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
    type Config: SaverConfig<Op::Sol,Scp,SolId,Obj,Opt,Op::SInfo,Op::Cod,Out>;
    fn init(&mut self);
    fn after_load(&mut self);
    fn save_state(&self, state: &Op::State, stop: &St, eval: &Eval);
    fn load(&self) -> Result<(St, Op, Eval), CheckpointError>;
    fn load_stop(&self) -> Result<St, CheckpointError>;
    fn load_optimizer(&self) -> Result<Op, CheckpointError>;
    fn load_evaluate(&self) -> Result<Eval, CheckpointError>;
    fn load_parameters(&self) -> Result<GlobalParameters, CheckpointError>;
}

#[cfg(feature = "mpi")]
pub trait DistCheckpointer<SolId, St, Obj, Opt, Out, Scp, Op, Eval>: Checkpointer<SolId, St, Obj, Opt, Out, Scp, Op, Eval>
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
    fn init(&mut self, rank:Rank);
    fn after_load(&mut self, rank:Rank);
    fn save_state(&self, state: &Op::State, stop: &St, eval: &Eval, rank:Rank);
    fn load(&self, rank:Rank) -> Result<(St, Op, Eval), CheckpointError>;
    fn load_stop(&self, rank:Rank) -> Result<St, CheckpointError>;
    fn load_optimizer(&self, rank:Rank) -> Result<Op, CheckpointError>;
    fn load_evaluate(&self, rank:Rank) -> Result<Eval, CheckpointError>;
    fn load_parameters(&self, rank:Rank) -> Result<GlobalParameters, CheckpointError>;
}