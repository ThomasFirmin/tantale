use crate::{
    domain::Domain,
    experiment::Evaluate,
    objective::{Codomain, LinkedOutcome, Outcome},
    optimizer::{ArcVecArc, Optimizer},
    saver::{CheckpointError, Saver},
    searchspace::Searchspace,
    solution::{Computed, Id},
    stop::Stop,
    GlobalParameters,
};

use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "mpi")]
use crate::saver::DistributedSaver;
#[cfg(feature = "mpi")]
use mpi::Rank;

/// [`NoSaver`] does nothing, and does not save anything.
#[derive(Default, Serialize, Deserialize)]
pub struct NoSaver {}

impl NoSaver {
    pub fn new() -> NoSaver {
        NoSaver {}
    }
}

impl<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>
    Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval> for NoSaver
where
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
    fn init(&mut self, _sp: &Scp, _cod: &Cod) {}
    fn after_load(&mut self, _sp: &Scp, _cod: &Cod) {}
    fn save_partial(
        &self,
        _obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, Op::SInfo>>,
        _opt: ArcVecArc<Computed<SolId, Opt, Cod, Out, Op::SInfo>>,
        _sp: Arc<Scp>,
        _cod: Arc<Cod>,
        _info: Arc<Op::Info>,
    ) {
    }

    fn save_codom(
        &self,
        _obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, Op::SInfo>>,
        _sp: Arc<Scp>,
        _cod: Arc<Cod>,
    ) {
    }

    fn save_out(&self, _lout: Vec<LinkedOutcome<Out, SolId, Obj, Op::SInfo>>, _sp: Arc<Scp>) {}

    fn save_state(&self, _sp: Arc<Scp>, _state: &Op::State, _stop: &St, _eval: &Eval) {}

    fn load(&self, _sp: &Scp, _cod: &Cod) -> Result<(St, Op, Eval), CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_stop(&self, _sp: &Scp, _cod: &Cod) -> Result<St, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_optimizer(&self, _sp: &Scp, _cod: &Cod) -> Result<Op, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_evaluate(&self, _sp: &Scp, _cod: &Cod) -> Result<Eval, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_parameters(&self, _sp: &Scp, _cod: &Cod) -> Result<GlobalParameters, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }
    fn clean(self) {}
}

#[cfg(feature = "mpi")]
impl<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>
    DistributedSaver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval> for NoSaver
where
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
    fn init(&mut self, _sp: &Scp, _cod: &Cod, _rank: Rank) {}
    fn after_load(&mut self, _sp: &Scp, _cod: &Cod, _rank: Rank) {}
    fn save_partial(
        &self,
        _obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, Op::SInfo>>,
        _opt: ArcVecArc<Computed<SolId, Opt, Cod, Out, Op::SInfo>>,
        _sp: Arc<Scp>,
        _cod: Arc<Cod>,
        _info: Arc<Op::Info>,
        _rank: Rank,
    ) {
    }

    fn save_codom(
        &self,
        _obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, Op::SInfo>>,
        _sp: Arc<Scp>,
        _cod: Arc<Cod>,
        _rank: Rank,
    ) {
    }

    fn save_out(
        &self,
        _lout: Vec<LinkedOutcome<Out, SolId, Obj, Op::SInfo>>,
        _sp: Arc<Scp>,
        _rank: Rank,
    ) {
    }

    fn save_state(&self, _sp: Arc<Scp>, _state: &Op::State, _stop: &St, _eval: &Eval, _rank: Rank) {
    }

    fn load(&self, _sp: &Scp, _cod: &Cod, _rank: Rank) -> Result<(St, Op, Eval), CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_stop(&self, _sp: &Scp, _cod: &Cod, _rank: Rank) -> Result<St, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_optimizer(&self, _sp: &Scp, _cod: &Cod, _rank: Rank) -> Result<Op, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_evaluate(&self, _sp: &Scp, _cod: &Cod, _rank: Rank) -> Result<Eval, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_parameters(
        &self,
        _sp: &Scp,
        _cod: &Cod,
        _rank: Rank,
    ) -> Result<GlobalParameters, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }
    
}
