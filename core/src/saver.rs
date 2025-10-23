use crate::{
    GlobalParameters, Onto, Optimizer, domain::Domain, experiment::Evaluate, objective::Outcome, optimizer::opt::{CBType, OBType, PBType}, searchspace::Searchspace, solution::Id, stop::Stop
};
use std::sync::Arc;

#[cfg(feature = "mpi")]
use mpi::Rank;

pub mod csvsaver;
pub use csvsaver::{BatchCSVWrite, CSVLeftRight, CSVSaver, CSVWritable};

pub mod nosaver;
pub use nosaver::NoSaver;

pub mod serror;
pub use serror::CheckpointError;

pub trait Saver<SolId, St, Obj, Opt, Out, Scp, Op, Eval>
where
    Self: Sized,
    SolId: Id,
    St: Stop,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom>,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    Eval: Evaluate,
{
    fn init(&mut self, sp: &Scp, cod: &Op::Cod);
    fn after_load(&mut self, sp: &Scp, cod: &Op::Cod);
    fn save_partial(&self, batch: &PBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>);
    fn save_partial_with_raw(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>);
    fn save_partial_with_comp(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>);
    fn save_info(&self, batch: &PBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>);
    fn save_info_with_raw(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>);
    fn save_info_with_comp(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>);
    fn save_out(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>);
    fn save_codom(
        &self,
        batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>,
        sp: Arc<Scp>,
        cod: Arc<Op::Cod>,
    );
    fn save_state(&self, sp: Arc<Scp>, state: &Op::State, stop: &St, eval: &Eval);
    fn load(&self, sp: &Scp, cod: &Op::Cod) -> Result<(St, Op, Eval), CheckpointError>;
    fn load_stop(&self, sp: &Scp, cod: &Op::Cod) -> Result<St, CheckpointError>;
    fn load_optimizer(&self, sp: &Scp, cod: &Op::Cod) -> Result<Op, CheckpointError>;
    fn load_evaluate(&self, sp: &Scp, cod: &Op::Cod) -> Result<Eval, CheckpointError>;
    fn load_parameters(&self, sp: &Scp, cod: &Op::Cod)
        -> Result<GlobalParameters, CheckpointError>;
    fn clean(self);
}

#[cfg(feature = "mpi")]
pub trait DistributedSaver<SolId, St, Obj, Opt, Out, Scp, Op, Eval>:
    Saver<SolId, St, Obj, Opt, Out, Scp, Op, Eval>
where
    Self: Sized,
    SolId: Id,
    St: Stop,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom>,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    Eval: Evaluate,
{
    fn init(&mut self, sp: &Scp, cod: &Op::Cod, rank: Rank);
    fn after_load(&mut self, sp: &Scp, cod: &Op::Cod, rank: Rank);
    fn save_partial(&self, batch: &Op::BType, sp: Arc<Scp>, rank: Rank);
    fn save_partial_with_raw(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>, rank:Rank);
    fn save_partial_with_comp(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>, rank:Rank);
    fn save_codom(
        &self,
        batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>,
        sp: Arc<Scp>,
        cod: Arc<Op::Cod>,
        rank: Rank,
    );
    fn save_info(&self, batch: &PBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>, rank: Rank);
    fn save_info_with_raw(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>, rank:Rank);
    fn save_info_with_comp(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>, rank:Rank);
    fn save_out(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>, sp: Arc<Scp>, rank: Rank);
    fn save_state(&self, sp: Arc<Scp>, state: &Op::State, stop: &St, eval: &Eval, rank: Rank);
    fn load(&self, sp: &Scp, cod: &Op::Cod, rank: Rank) -> Result<(St, Op, Eval), CheckpointError>;
    fn load_stop(&self, sp: &Scp, cod: &Op::Cod, rank: Rank) -> Result<St, CheckpointError>;
    fn load_optimizer(&self, sp: &Scp, cod: &Op::Cod, rank: Rank) -> Result<Op, CheckpointError>;
    fn load_evaluate(&self, sp: &Scp, cod: &Op::Cod, rank: Rank) -> Result<Eval, CheckpointError>;
    fn load_parameters(
        &self,
        sp: &Scp,
        cod: &Op::Cod,
        rank: Rank,
    ) -> Result<GlobalParameters, CheckpointError>;
}
