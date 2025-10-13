use crate::{
    GlobalParameters, Optimizer, domain::Domain, experiment::Evaluate, objective::{Codomain, Outcome}, optimizer::{opt::{CBType, OBType, PBType}}, searchspace::Searchspace, solution::Id, stop::Stop
};
use std::sync::Arc;

#[cfg(feature = "mpi")]
use mpi::Rank;

pub mod csvsaver;
pub use csvsaver::{CSVLeftRight, CSVSaver, CSVWritable, BatchCSVWrite};

pub mod nosaver;
pub use nosaver::NoSaver;

pub mod serror;
pub use serror::CheckpointError;

#[macro_export]
macro_rules! load {
    ($experiment: ident, $optimizer : ident, $stop : ident | $searchspace : expr, $objective : expr , $saver : expr) => {
        $experiment::<_,_,$optimizer, $stop, _, _, _, _, _, _, _>::load($searchspace, $objective, $saver)
    };
    ($experiment: ident, $optimizer : ident, $stop : ident | $process : expr, $searchspace : expr, $objective : expr , $saver : expr) => {
        $experiment::<_,_,$optimizer, $stop, _, _, _, _, _, _, _>::load($process, $searchspace, $objective, $saver)
    };
}

pub trait Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>
where
    Self: Sized,
    SolId: Id,
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Cod, Out, Scp>,
    Eval: Evaluate,
{
    fn init(&mut self, sp: &Scp, cod: &Cod);
    fn after_load(&mut self, sp: &Scp, cod: &Cod);
    fn save_partial(
        &self,
        batch : &PBType<Op,SolId,Obj,Opt,Cod,Out,Scp>,
        sp: Arc<Scp>,
    );
    fn save_info(&self, batch: &PBType<Op,SolId,Obj,Opt,Cod,Out,Scp>, sp: Arc<Scp>);
    fn save_out(&self, batch: &OBType<Op,SolId,Obj,Opt,Cod,Out,Scp>, sp: Arc<Scp>);
    fn save_codom(
        &self,
        batch: &CBType<Op,SolId,Obj,Opt,Cod,Out,Scp>,
        sp: Arc<Scp>,
        cod: Arc<Cod>,
    );
    fn save_state(&self, sp: Arc<Scp>, state: &Op::State, stop: &St, eval: &Eval);
    fn load(&self, sp: &Scp, cod: &Cod) -> Result<(St, Op, Eval), CheckpointError>;
    fn load_stop(&self, sp: &Scp, cod: &Cod) -> Result<St, CheckpointError>;
    fn load_optimizer(&self, sp: &Scp, cod: &Cod) -> Result<Op, CheckpointError>;
    fn load_evaluate(&self, sp: &Scp, cod: &Cod) -> Result<Eval, CheckpointError>;
    fn load_parameters(&self, sp: &Scp, cod: &Cod) -> Result<GlobalParameters, CheckpointError>;
    fn clean(self);
}

#[cfg(feature = "mpi")]
pub trait DistributedSaver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>:
    Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>
where
    Self: Sized,
    SolId: Id,
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Cod, Out, Scp>,
    Eval: Evaluate,
{
    fn init(&mut self, sp: &Scp, cod: &Cod, rank: Rank);
    fn after_load(&mut self, sp: &Scp, cod: &Cod, rank: Rank);
    fn save_partial(
        &self,
        batch : &Op::BType,
        sp: Arc<Scp>,
        rank: Rank,
    );
    fn save_codom(
        &self,
        batch: &CBType<Op,SolId,Obj,Opt,Cod,Out,Scp>,
        sp: Arc<Scp>,
        cod: Arc<Cod>,
        rank: Rank,
    );
    fn save_info(&self, batch: &PBType<Op,SolId,Obj,Opt,Cod,Out,Scp>, sp: Arc<Scp>, rank:Rank);
    fn save_out(
        &self,
        batch: &OBType<Op,SolId,Obj,Opt,Cod,Out,Scp>,
        sp: Arc<Scp>,
        rank: Rank,
    );
    fn save_state(&self, sp: Arc<Scp>, state: &Op::State, stop: &St, eval: &Eval, rank: Rank);
    fn load(&self, sp: &Scp, cod: &Cod, rank: Rank) -> Result<(St, Op, Eval), CheckpointError>;
    fn load_stop(&self, sp: &Scp, cod: &Cod, rank: Rank) -> Result<St, CheckpointError>;
    fn load_optimizer(&self, sp: &Scp, cod: &Cod, rank: Rank) -> Result<Op, CheckpointError>;
    fn load_evaluate(&self, sp: &Scp, cod: &Cod, rank: Rank) -> Result<Eval, CheckpointError>;
    fn load_parameters(
        &self,
        sp: &Scp,
        cod: &Cod,
        rank: Rank,
    ) -> Result<GlobalParameters, CheckpointError>;
}
