use crate::{
    domain::Domain,
    experiment::{Evaluate, SeqEvaluate, SeqEvaluator, Runable},
    objective::{Codomain, Objective, Outcome},
    optimizer::opt::{SequentialOptimizer},
    saver::Saver,
    searchspace::Searchspace,
    solution::SId,
    stop::{ExpStep, Stop},
};

use std::{marker::PhantomData, sync::{Arc, Mutex}};

type EvalType<Obj, Opt, Info, SInfo> = Option<SeqEvaluator<SId, Obj, Opt, Info, SInfo>>;

pub struct Experiment<Eval, Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Eval: Evaluate,
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
        Eval,
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
    _eval: PhantomData<Eval>,
    _domobj: PhantomData<Obj>,
    _domopt: PhantomData<Opt>,
    _codom: PhantomData<Cod>,
    _out: PhantomData<Out>,
}

impl<Eval, Scp, Op, St, Sv, Obj, Opt, Out, Cod> Experiment<Eval, Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Eval:Evaluate,
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
        Eval,
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
            _eval: PhantomData,
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
        SeqEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    > for Experiment<SeqEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>, Scp, Op, St, Sv, Obj, Opt, Out, Cod>
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
        SeqEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
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
                SeqEvaluator::new(sobj.clone(), sopt.clone(), info)
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
            eval.evaluate(&mut eval, ob.clone(), st.clone());

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
            <SeqEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo> as SeqEvaluate<
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