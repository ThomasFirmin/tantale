use crate::{
    ArcVecArc, Computed, Id, LinkedOutcome, OptInfo, Partial, SolInfo, Solution, domain::Domain, experiment::{
        DistEvaluate, Evaluate, Runable,
    }, objective::{Codomain, Objective, Outcome}, optimizer::opt::{SequentialOptimizer,SolPairs}, saver::Saver, searchspace::Searchspace, solution::SId, stop::{ExpStep, Stop}
};

use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap, marker::PhantomData, sync::{Arc, Mutex}
};

#[cfg(feature="mpi")]
use crate::{
    experiment::mpi::{tools::MPIProcess, utils::{OMessage, SolPair, VecArcComputed,fill_workers,send_to_worker}}
};
#[cfg(feature="mpi")]
use mpi::{traits::{Communicator,Source}};
#[cfg(feature="mpi")]
use bincode::serde::Compat;

type EvalType<Obj, Opt, Info, SInfo> = Option<Evaluator<SId, Obj, Opt, Info, SInfo>>;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Evaluator<SolId, Obj, Opt, Info, SInfo>
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

impl<SolId, Obj, Opt, Info, SInfo> Evaluator<SolId, Obj, Opt, Info, SInfo>
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
        Evaluator {
            in_obj,
            in_opt,
            info,
            idx: 0,
        }
    }
}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    Evaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Objective<Obj, Cod, Out>>
    for Evaluator<SolId, Obj, Opt, Info, SInfo>
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
        let mut result_obj = Vec::new();
        let mut result_opt = Vec::new();
        let mut result_out = Vec::new();
        let mut st = stop.lock().unwrap();

        let mut i = self.idx;
        let length = self.in_obj.len();
        while i < length && !st.stop() {
            let sobj = self.in_obj[i].clone();
            let sopt = self.in_opt[i].clone();
            let (cod, out) = ob.compute(sobj.get_x().as_ref());
            result_obj.push(Arc::new(Computed::new(sobj.clone(), cod.clone())));
            result_opt.push(Arc::new(Computed::new(sopt.clone(), cod.clone())));
            result_out.push(LinkedOutcome::new(out.clone(), sobj.clone()));
            st.update(ExpStep::Distribution);
            i += 1
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

#[cfg(feature="mpi")]
impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    DistEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Objective<Obj, Cod, Out>>
    for Evaluator<SolId, Obj, Opt, Info, SInfo>
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
        proc:&MPIProcess,
        ob: Arc<Objective<Obj, Cod, Out>>,
        stop: Arc<Mutex<St>>,
    ) -> (
        SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
        Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
    ) {
        // Bytes encoding config
        let config = bincode::config::standard();
        // [1..SIZE] because of master process
        let mut idle_process: Vec<i32> = (1..proc.size).collect();
        let mut waiting: HashMap<SolId, SolPair<SolId, Obj, Opt, SInfo>> = HashMap::new();
        let mut i = fill_workers(
            &proc,
            &mut idle_process,
            stop.clone(),
            self.in_obj.clone(),
            self.in_opt.clone(),
            self.idx,
            &mut waiting,
            config,
        );

        // Main variables
        let length = self.in_obj.len();
        //Results
        let mut result_obj: VecArcComputed<SolId, Obj, Cod, Out, SInfo> = Vec::new();
        let mut result_opt: VecArcComputed<SolId, Opt, Cod, Out, SInfo> = Vec::new();
        let mut result_out: Vec<LinkedOutcome<Out, SolId, Obj, SInfo>> = Vec::new();

        // Recv / sendv loop
        while !waiting.is_empty() {
            let (bytes, status): (Vec<u8>, _) = proc.world.any_process().receive_vec();
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
            if !stop.lock().unwrap().stop() && i < length {
                let has_idl = send_to_worker(
                    &proc.world,
                    &mut idle_process,
                    config,
                    self.in_obj[i].clone(),
                    self.in_opt[i].clone(),
                    &mut waiting,
                );
                if has_idl {
                    stop.lock().unwrap().update(ExpStep::Distribution);
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


pub struct Experiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
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

impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod> Experiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
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
        saver.init(&searchspace, objective.get_codomain().as_ref());
        Experiment {
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
        Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    > for Experiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo> + Send + Sync,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    fn run(mut self) {
        let sp = Arc::new(self.searchspace);
        let ob = Arc::new(self.objective);
        let cod = ob.get_codomain();
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let (sobj, sopt, info) = self.optimizer.first_step(sp.clone());
                Evaluator::new(sobj.clone(), sopt.clone(), info)
            }
        };

        let (mut sobj, mut sopt, mut info): (_, _, Arc<Op::Info>);
        'main: loop {
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Evaluation);
                self.saver
                    .save_state(sp.clone(), self.optimizer.get_state(), &st, &eval);
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            let ((cobj, copt), cout) =
                <Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo> as Evaluate<
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

            // Saver part
            self.saver.save_partial(
                cobj.clone(),
                copt.clone(),
                sp.clone(),
                cod.clone(),
                eval.info.clone(),
            );
            self.saver.save_out(cout, sp.clone());
            self.saver.save_codom(cobj.clone(), sp.clone(), cod.clone());

            if st.lock().unwrap().stop() {
                self.saver.save_state(
                    sp.clone(),
                    self.optimizer.get_state(),
                    &st.lock().unwrap(),
                    &eval,
                );
                break 'main;
            };
            (sobj, sopt, info) = self.optimizer.step((cobj, copt), sp.clone());
            <Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo> as Evaluate<
                St,
                Obj,
                Opt,
                Out,
                Cod,
                Op::Info,
                Op::SInfo,
                SId,
                Objective<Obj, Cod, Out>,
            >>::update(&mut eval, sobj.clone(), sopt.clone(), info.clone());
            st.lock().unwrap().update(ExpStep::Optimization);
            if st.lock().unwrap().stop() {
                self.saver.save_state(
                    sp.clone(),
                    self.optimizer.get_state(),
                    &st.lock().unwrap(),
                    &eval,
                );
                break 'main;
            };
        }
    }

    fn load(searchspace: Scp, objective: Objective<Obj, Cod, Out>, mut saver: Sv) -> Self {
        let (stop, optimizer, evaluator) = saver
            .load(&searchspace, objective.get_codomain().as_ref())
            .unwrap();
        Saver::after_load(
            &mut saver,
            &searchspace,
            objective.get_codomain().as_ref(),
        );
        Experiment {
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