use crate::{
    GlobalParameters,
    Optimizer,
    domain::Domain,
    experiment::Evaluate,
    objective::{Codomain, FuncWrapper, LinkedOutcome, Outcome},
    optimizer::ArcVecArc,
    searchspace::Searchspace,
    solution::{Computed, Id},
    stop::Stop
};
use std::sync::Arc;

#[cfg(feature = "mpi")]
use mpi::Rank;

pub mod csvsaver;
pub use csvsaver::{CSVLeftRight, CSVSaver, CSVWritable};

pub mod serror;
pub use serror::CheckpointError;

#[macro_export]
macro_rules! load {
    ($experiment: ident, $optimizer : ident, $stop : ident | $searchspace : expr, $objective : expr , $saver : expr) => {
        $experiment::<_, $optimizer, $stop, _, _, _, _, _>::load($searchspace, $objective, $saver)
    };
    ($experiment: ident, $optimizer : ident, $stop : ident | $searchspace : expr, $objective : expr , $saver : expr) => {
        $experiment::<_, $optimizer, $stop, _, _, _, _, _, _>::load($searchspace, $objective, $saver)
    };
    ($rank : expr, $experiment: ident, $optimizer : ident, $stop : ident | $searchspace : expr, $objective : expr , $saver : expr) => {
        $experiment::<_, $optimizer, $stop, _, _, _, _, _>::load($searchspace, $objective, $saver)
    };
}

pub trait Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval, FnWrap>
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
    Eval: Evaluate<St, Obj, Opt, Out, Cod, Op::Info, Op::SInfo, SolId, FnWrap>,
    FnWrap: FuncWrapper,
{
    fn init(&mut self, sp: &Scp, cod: &Cod);
    fn save_partial(
        &self,
        obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, Op::SInfo>>,
        opt: ArcVecArc<Computed<SolId, Opt, Cod, Out, Op::SInfo>>,
        sp: Arc<Scp>,
        cod: Arc<Cod>,
        info: Arc<Op::Info>,
    );
    fn save_codom(
        &self,
        obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, Op::SInfo>>,
        sp: Arc<Scp>,
        cod: Arc<Cod>,
    );
    fn save_out(&self, lout: Vec<LinkedOutcome<Out, SolId, Obj, Op::SInfo>>, sp: Arc<Scp>);
    fn save_state(&self, sp: Arc<Scp>, state: &Op::State, stop: &St, eval: &Eval);
    fn load(&self, sp: &Scp, cod: &Cod) -> Result<(St, Op, Eval), CheckpointError>;
    fn load_stop(&self, sp: &Scp, cod: &Cod) -> Result<St, CheckpointError>;
    fn load_optimizer(&self, sp: &Scp, cod: &Cod) -> Result<Op, CheckpointError>;
    fn load_evaluate(&self, sp: &Scp, cod: &Cod) -> Result<Eval, CheckpointError>;
    fn load_parameters(&self, sp: &Scp, cod: &Cod) -> Result<GlobalParameters, CheckpointError>;
    fn clean(self);
}

#[cfg(feature = "mpi")]
pub trait DistributedSaver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval, FnWrap>
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
    Eval: Evaluate<St, Obj, Opt, Out, Cod, Op::Info, Op::SInfo, SolId, FnWrap>,
    FnWrap: FuncWrapper,
{
    fn init(&mut self, sp: &Scp, cod: &Cod, rank:Rank);
    fn save_partial(
        &self,
        obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, Op::SInfo>>,
        opt: ArcVecArc<Computed<SolId, Opt, Cod, Out, Op::SInfo>>,
        sp: Arc<Scp>,
        cod: Arc<Cod>,
        info: Arc<Op::Info>, 
        rank:Rank
    );
    fn save_codom(
        &self,
        obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, Op::SInfo>>,
        sp: Arc<Scp>,
        cod: Arc<Cod>, 
        rank:Rank
    );
    fn save_out(&self, lout: Vec<LinkedOutcome<Out, SolId, Obj, Op::SInfo>>, sp: Arc<Scp>, rank:Rank);
    fn save_state(&self, sp: Arc<Scp>, state: &Op::State, stop: &St, eval: &Eval, rank:Rank);
    fn load(&self, sp: &Scp, cod: &Cod, rank:Rank) -> Result<(St, Op, Eval), CheckpointError>;
    fn load_stop(&self, sp: &Scp, cod: &Cod, rank:Rank) -> Result<St, CheckpointError>;
    fn load_optimizer(&self, sp: &Scp, cod: &Cod, rank:Rank) -> Result<Op, CheckpointError>;
    fn load_evaluate(&self, sp: &Scp, cod: &Cod, rank:Rank) -> Result<Eval, CheckpointError>;
    fn load_parameters(&self, sp: &Scp, cod: &Cod, rank:Rank) -> Result<GlobalParameters, CheckpointError>;
}