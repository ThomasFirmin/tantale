use crate::{
    Fidelity, domain::Domain, experiment::{
        Evaluate, Runable, sequential::fidelity::seqevaluator::{Evaluator, ParEvaluator}
    }, objective::{Objective, Outcome}, optimizer::opt::SequentialOptimizer, saver::Saver, searchspace::Searchspace, solution::SId, stop::{ExpStep, Stop}
};

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

type EvalType<Obj, Opt, Info, SInfo, Out> = Option<Evaluator<SId, Obj, Opt, Info, SInfo, Out>>;
type ParEvalType<Obj, Opt, Info, SInfo, Out> = Option<ParEvaluator<SId, Obj, Opt, Info, SInfo, Out>>;

pub struct Experiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: Saver<SId, St, Obj, Opt, Cod, Out, Scp, Op, Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo, Out>>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Fidelity<Out>,
{
    pub searchspace: Scp,
    pub objective: Objective<Obj, Cod, Out>,
    pub optimizer: Op,
    pub stop: St,
    pub saver: Sv,
    evaluator: EvalType<Obj, Opt, Op::Info, Op::SInfo,Out>,
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
    Sv: Saver<SId, St, Obj, Opt, Cod, Out, Scp, Op, Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo,Out>>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Fidelity<Out>,
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
    Runable<SId, Scp, Op, St, Sv, Obj, Opt, Out, Cod, Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo, Out>>
    for Experiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo> + Send + Sync,
    Sv: Saver<SId, St, Obj, Opt, Cod, Out, Scp, Op, Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo, Out>>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Fidelity<Out>,
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
                <Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo, Out> as Evaluate<
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    Op::Info,
                    Op::SInfo,
                    SId,
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
            <Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo,Out> as Evaluate<
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    Op::Info,
                    Op::SInfo,
                    SId,
                >>::update(&mut eval, sobj.clone(),sopt.clone(),info.clone());
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
    }

    fn load(searchspace: Scp, objective: Objective<Obj, Cod, Out>, saver: Sv) -> Self {
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

pub struct ParExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop + Send + Sync,
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
            ParEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, Out>,
        > + Send
        + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Fidelity<Out> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::State: Send + Sync,
{
    pub searchspace: Scp,
    pub objective: Objective<Obj, Cod, Out>,
    pub optimizer: Op,
    pub stop: St,
    pub saver: Sv,
    evaluator: ParEvalType<Obj, Opt, Op::Info, Op::SInfo, Out>,
    _domobj: PhantomData<Obj>,
    _domopt: PhantomData<Opt>,
    _codom: PhantomData<Cod>,
    _out: PhantomData<Out>,
}

impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod> ParExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop + Send + Sync,
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
            ParEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, Out>,
        > + Send
        + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Fidelity<Out> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::State: Send + Sync,
{
    pub fn new(
        searchspace: Scp,
        objective: Objective<Obj, Cod, Out>,
        optimizer: Op,
        stop: St,
        mut saver: Sv,
    ) -> Self {
        saver.init(&searchspace, objective.get_codomain().as_ref());
        ParExperiment {
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
        ParEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, Out>,
    > for ParExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop + Send + Sync,
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
            ParEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, Out>,
        > + Send
        + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Fidelity<Out> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::State: Send + Sync,
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
                ParEvaluator::new(sobj.clone(), sopt.clone(), info)
            }
        };

        let (mut sobj, mut sopt, mut info): (_, _, Arc<Op::Info>);
        loop {
            {
                let mut st = st.lock().unwrap();
                self.saver
                    .save_state(sp.clone(), self.optimizer.get_state(), &st, &eval);
                st.update(ExpStep::Evaluation);
                if st.stop() {
                    break;
                };
            }

            // Arc copy of data to send to evaluator thread.
            let ((cobj, copt), cout) =
                <ParEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo,Out> as Evaluate<
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    Op::Info,
                    Op::SInfo,
                    SId,
                >>::evaluate(&mut eval, ob.clone(), st.clone());

            // Saver part
            let cobj1 = cobj.clone();
            let copt1 = copt.clone();
            let sp1 = sp.clone();
            let cod1 = cod.clone();
            let info1 = eval.info.clone();
            let cobj2 = cobj.clone();
            let sp2 = sp.clone();
            let cod2 = cod.clone();
            rayon::join(
                || {
                    let _ = &self.saver.save_partial(cobj1, copt1, sp1, cod1, info1);
                },
                || {
                    let _ = &self.saver.save_out(cout, sp2.clone());
                    let _ = &self.saver.save_codom(cobj2, sp2, cod2);
                },
            );

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
            <ParEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo,Out> as Evaluate<
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    Op::Info,
                    Op::SInfo,
                    SId,
                >>::update(&mut eval, sobj.clone(),sopt.clone(),info.clone());
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
    }

    fn load(searchspace: Scp, objective: Objective<Obj, Cod, Out>, saver: Sv) -> Self {
        let (stop, optimizer, evaluator) = saver
            .load(&searchspace, objective.get_codomain().as_ref())
            .unwrap();
        ParExperiment {
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
