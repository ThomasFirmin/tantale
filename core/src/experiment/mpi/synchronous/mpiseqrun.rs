use crate::{
    ArcVecArc, Codomain, Computed, Domain, Id, LinkedOutcome, MPI_RANK, MPI_SIZE, MPI_UNIVERSE, MPI_WORLD, Objective, OptInfo, Outcome, Partial, SId, Searchspace, SolInfo, experiment::{
        Evaluate, Runable, mpi::utils::{
            OMessage, SolPair, VecArcComputed, fill_workers, send_to_worker
        }
    }, optimizer::opt::{SequentialOptimizer, SolPairs}, saver::DistributedSaver, stop::{ExpStep, Stop}
};

use bincode::{self, serde::Compat};
use mpi::{
    traits::{Communicator, Source},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap, marker::PhantomData, sync::{Arc, Mutex}
};

type EvalType<Obj, Opt, Info, SInfo> = Option<MPIEvaluator<SId, Obj, Opt, Info, SInfo>>;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct MPIEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    pub in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    pub info: Arc<Info>,
    idx: usize,
}

impl<SolId, Obj, Opt, Info, SInfo> MPIEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
{
    pub fn new(
        in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) -> Self {
        MPIEvaluator {
            in_obj,
            in_opt,
            info,
            idx: 0,
        }
    }
}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    Evaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Objective<Obj, Cod, Out>>
    for MPIEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Objective<Obj, Cod, Out>>,
        stop: Arc<Mutex<St>>,
    ) -> (
        SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
        Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
    ) {
        // Bytes encoding config
        let config = bincode::config::standard();
        // [1..SIZE] because of master process
        let mut idle_process: Vec<i32> = (1..*MPI_SIZE.get().unwrap()).collect();
        let mut waiting: HashMap<SolId, SolPair<SolId, Obj, Opt, SInfo>> = HashMap::new();
        let mut i = fill_workers(
            &mut idle_process,
            stop.clone(),
            self.in_obj.clone(),
            self.in_opt.clone(),
            self.idx,
            &mut waiting,
            config,
        );

        // Main variables
        let world = MPI_WORLD.get().unwrap();
        let length = self.in_obj.len();
        //Results
        let mut result_obj: VecArcComputed<SolId, Obj, Cod, Out, SInfo> = Vec::new();
        let mut result_opt: VecArcComputed<SolId, Opt, Cod, Out, SInfo> = Vec::new();
        let mut result_out: Vec<LinkedOutcome<Out, SolId, Obj, SInfo>> = Vec::new();

        // Recv / sendv loop
        while !waiting.is_empty() {
            let (bytes, status): (Vec<u8>, _) = world.any_process().receive_vec();
            idle_process.push(status.source_rank());
            let (bytes, _): (Compat<OMessage<SolId, Out>>, _) =
                bincode::decode_from_slice(bytes.as_slice(), config).unwrap();
            let msg = bytes.0;
            let id = msg.0;
            let out = msg.1;
            let cod = Arc::new(ob.codomain.get_elem(&out));
            let out = Arc::new(out);
            let (sobj, sopt) = waiting.remove(&id).unwrap();
            result_obj.push(Arc::new(Computed::new(sobj.clone(), cod.clone())));
            result_opt.push(Arc::new(Computed::new(sopt.clone(), cod)));
            result_out.push(LinkedOutcome::new(out.clone(), sobj.clone()));
            stop.lock().unwrap().update(ExpStep::Distribution);
            if !stop.lock().unwrap().stop() && i < length {
                let has_idl = send_to_worker(
                    world,
                    &mut idle_process,
                    config,
                    self.in_obj[i].clone(),
                    self.in_opt[i].clone(),
                    &mut waiting,
                );
                if has_idl {
                    i += 1;
                }
            }
        }
        // For saving in case of early stopping before full evaluation of all elements
        self.idx = i;
        ((Arc::new(result_obj), Arc::new(result_opt)), result_out)
    }

    fn update(
        &mut self,
        obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) {
        self.in_obj = obj;
        self.in_opt = opt;
        self.info = info;
        self.idx = 0;
    }
}

pub struct MPIExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: DistributedSaver<
        SId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        MPIEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    pub searchspace: Scp,
    pub objective: Objective<Obj, Cod, Out>,
    pub optimizer: Op,
    pub stop: St,
    pub saver: Sv,
    evaluator: EvalType<Obj, Opt, Op::Info, Op::SInfo>,
    _domobj: PhantomData<Obj>,
    _domopt: PhantomData<Opt>,
    _codom: PhantomData<Cod>,
    _out: PhantomData<Out>,
}

impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod> MPIExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: DistributedSaver<
        SId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        MPIEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    pub fn new(
        searchspace: Scp,
        objective: Objective<Obj, Cod, Out>,
        optimizer: Op,
        stop: St,
        mut saver: Sv,
    ) -> Self {
        DistributedSaver::init(
            &mut saver,
            &searchspace,
            objective.get_codomain().as_ref(),
            *MPI_RANK.get().unwrap(),
        );
        MPIExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            saver,
            evaluator: None,
            _domobj: PhantomData,
            _domopt: PhantomData,
            _codom: PhantomData,
            _out: PhantomData,
        }
    }
}

impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
    Runable<
        SId,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
        Cod,
        MPIEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    > for MPIExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: DistributedSaver<
        SId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        MPIEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    fn run(mut self) {
        if MPI_UNIVERSE.get().is_none() {
            panic!("The MPI Universe is not initialized.")
        }
        let rank = *MPI_RANK.get().unwrap();
        let sp = Arc::new(self.searchspace);
        let ob = Arc::new(self.objective);
        let cod = ob.get_codomain();
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let (sobj, sopt, info) = self.optimizer.first_step(sp.clone());
                MPIEvaluator::new(sobj.clone(), sopt.clone(), info)
            }
        };

        let (mut sobj, mut sopt, mut info): (_, _, Arc<Op::Info>);
        'main: loop {
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Evaluation);
                DistributedSaver::save_state(
                    &self.saver,
                    sp.clone(),
                    self.optimizer.get_state(),
                    &st,
                    &eval,
                    rank,
                );
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            let ((cobj, copt), cout) =
                <MPIEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo> as Evaluate<
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    Op::Info,
                    Op::SInfo,
                    SId,
                    Objective<Obj, Cod, Out>,
                >>::evaluate(&mut eval, ob.clone(), st.clone());

            // DistributedSaver part
            DistributedSaver::save_partial(
                &self.saver,
                cobj.clone(),
                copt.clone(),
                sp.clone(),
                cod.clone(),
                eval.info.clone(),
                rank,
            );
            DistributedSaver::save_out(&self.saver, cout, sp.clone(), rank);
            DistributedSaver::save_codom(&self.saver, cobj.clone(), sp.clone(), cod.clone(), rank);
            if st.lock().unwrap().stop() {
                DistributedSaver::save_state(
                    &self.saver,
                    sp.clone(),
                    self.optimizer.get_state(),
                    &st.lock().unwrap(),
                    &eval,
                    rank,
                );
                break 'main;
            };
            (sobj, sopt, info) = self.optimizer.step((cobj, copt), sp.clone());
            eval = MPIEvaluator::new(sobj.clone(), sopt.clone(), info);
            st.lock().unwrap().update(ExpStep::Optimization);
            if st.lock().unwrap().stop() {
                DistributedSaver::save_state(
                    &self.saver,
                    sp.clone(),
                    self.optimizer.get_state(),
                    &st.lock().unwrap(),
                    &eval,
                    rank,
                );
                break 'main;
            };
        }
        let world = MPI_WORLD.get().unwrap();
        world.abort(42)
    }

    fn load(searchspace: Scp, objective: Objective<Obj, Cod, Out>, saver: Sv) -> Self {
        let rank = *MPI_RANK.get().unwrap();
        let (stop, optimizer, evaluator) =
            DistributedSaver::load(&saver, &searchspace, objective.get_codomain().as_ref(),rank)
                .unwrap();
        MPIExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            saver,
            evaluator: Some(evaluator),
            _domobj: PhantomData,
            _domopt: PhantomData,
            _codom: PhantomData,
            _out: PhantomData,
        }
    }
}