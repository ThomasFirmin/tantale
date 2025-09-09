use crate::{
    domain::Domain,
    experiment::{
        sequential::seqevaluator::{Evaluator, ParEvaluator},
        Evaluate, Runable,
    },
    objective::{Codomain, Objective, Outcome},
    optimizer::opt::SequentialOptimizer,
    saver::Saver,
    searchspace::Searchspace,
    solution::DistSId,
    stop::{ExpStep, Stop},
};

use mpi::{
    environment::Universe,
    topology::{Rank, SimpleCommunicator},
};
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

type EvalType<Obj, Opt, Info, SInfo> = Option<Evaluator<DistSId, Obj, Opt, Info, SInfo>>;

pub struct Experiment<Scp, Ob, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<DistSId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<DistSId, Obj, Opt, Op::SInfo>,
    Ob: Objective<Obj, Cod, Out>,
    Sv: Saver<
        DistSId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        Ob,
        Evaluator<DistSId, Obj, Opt, Op::Info, Op::SInfo>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    pub searchspace: Scp,
    pub objective: Ob,
    pub optimizer: Op,
    pub stop: St,
    pub saver: Sv,
    evaluator: EvalType<Obj, Opt, Op::Info, Op::SInfo>,
    universe: Universe,
    world: SimpleCommunicator,
    wsize: Rank,
    rank: Rank,
    _domobj: PhantomData<Obj>,
    _domopt: PhantomData<Opt>,
    _codom: PhantomData<Cod>,
    _out: PhantomData<Out>,
}

impl<Scp, Ob, Op, St, Sv, Obj, Opt, Out, Cod> Experiment<Scp, Ob, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<DistSId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<DistSId, Obj, Opt, Op::SInfo>,
    Ob: Objective<Obj, Cod, Out>,
    Sv: Saver<
        DistSId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        Ob,
        Evaluator<DistSId, Obj, Opt, Op::Info, Op::SInfo>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    pub fn new(searchspace: Scp, objective: Ob, optimizer: Op, stop: St, mut saver: Sv) -> Self {
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
    pub fn launch_worker(&self) {}
    pub fn launch_master(&self) {}
}

impl<Scp, Ob, Op, St, Sv, Obj, Opt, Out, Cod>
    Runable<
        DistSId,
        Scp,
        Ob,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
        Cod,
        Evaluator<DistSId, Obj, Opt, Op::Info, Op::SInfo>,
    > for Experiment<Scp, Ob, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<DistSId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<DistSId, Obj, Opt, Op::SInfo> + Send + Sync,
    Ob: Objective<Obj, Cod, Out>,
    Sv: Saver<
        DistSId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        Ob,
        Evaluator<DistSId, Obj, Opt, Op::Info, Op::SInfo>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    fn run(mut self) {
        if self.rank == 0 {
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
            loop {
                {
                    let mut st = st.lock().unwrap();
                    st.update(ExpStep::Evaluation);
                    self.saver
                        .save_state(sp.clone(), self.optimizer.get_state(), &st, &eval);
                    if st.stop() {
                        break;
                    };
                }

                // Arc copy of data to send to evaluator thread.
                let ((cobj, copt), cout) =
                    <Evaluator<DistSId, Obj, Opt, Op::Info, Op::SInfo> as Evaluate<
                        Ob,
                        St,
                        Obj,
                        Opt,
                        Out,
                        Cod,
                        Op::Info,
                        Op::SInfo,
                        DistSId,
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
                    break;
                };
                (sobj, sopt, info) = self.optimizer.step((cobj, copt), sp.clone());
                eval = Evaluator::new(sobj.clone(), sopt.clone(), info);
                st.lock().unwrap().update(ExpStep::Optimization);
                if st.lock().unwrap().stop() {
                    self.saver.save_state(
                        sp.clone(),
                        self.optimizer.get_state(),
                        &st.lock().unwrap(),
                        &eval,
                    );
                    break;
                };
            }
        } else {
            self.launch_worker();
        }
    }

    fn load(searchspace: Scp, objective: Ob, saver: Sv) -> Self {
        let (stop, optimizer, evaluator) = saver
            .load(&searchspace, objective.get_codomain().as_ref())
            .unwrap();
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
