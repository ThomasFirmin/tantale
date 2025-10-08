use crate::{
    domain::Domain,
    experiment::{Evaluate, FidEvaluator, FidThrEvaluator, MonoEvaluate, Runable, ThrEvaluate},
    objective::{outcome::FuncState, Outcome, Stepped},
    optimizer::opt::Optimizer,
    saver::Saver,
    searchspace::Searchspace,
    solution::SId,
    stop::{ExpStep, Stop},
    Fidelity,
};

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};
pub struct FidExperiment<Eval, Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
where
    Eval: Evaluate,
    Op: Optimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: Saver<SId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Fidelity<Out>,
    FnState: FuncState,
{
    pub searchspace: Scp,
    pub objective: Stepped<Obj, Cod, Out, FnState>,
    pub optimizer: Op,
    pub stop: St,
    pub saver: Sv,
    evaluator: Option<Eval>,
    _domobj: PhantomData<Obj>,
    _domopt: PhantomData<Opt>,
    _codom: PhantomData<Cod>,
    _out: PhantomData<Out>,
}

impl<Eval, Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
    FidExperiment<Eval, Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
where
    Eval: Evaluate,
    Op: Optimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: Saver<SId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Fidelity<Out>,
    FnState: FuncState,
{
    pub fn new(
        searchspace: Scp,
        objective: Stepped<Obj, Cod, Out, FnState>,
        optimizer: Op,
        stop: St,
        mut saver: Sv,
    ) -> Self {
        saver.init(&searchspace, objective.get_codomain().as_ref());
        FidExperiment {
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

impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
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
        FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
        Stepped<Obj, Cod, Out, FnState>,
    >
    for FidExperiment<
        FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
        Cod,
        FnState,
    >
where
    Op: Optimizer<SId, Obj, Opt, Cod, Out, Scp>,
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
        FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Fidelity<Out>,
    FnState: FuncState,
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
                FidEvaluator::new(sobj.clone(), sopt.clone(), info)
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
                <FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState> as MonoEvaluate<
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    Op::Info,
                    Op::SInfo,
                    SId,
                    Stepped<Obj, Cod, Out, FnState>,
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
            <FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState> as MonoEvaluate<
                St,
                Obj,
                Opt,
                Out,
                Cod,
                Op::Info,
                Op::SInfo,
                SId,
                Stepped<Obj, Cod, Out, FnState>,
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

    fn load(searchspace: Scp, objective: Stepped<Obj, Cod, Out, FnState>, mut saver: Sv) -> Self {
        let (stop, optimizer, evaluator) = saver
            .load(&searchspace, objective.get_codomain().as_ref())
            .unwrap();
        Saver::after_load(&mut saver, &searchspace, objective.get_codomain().as_ref());
        FidExperiment {
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

impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
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
        FidThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
        Stepped<Obj, Cod, Out, FnState>,
    >
    for FidExperiment<
        FidThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
        Cod,
        FnState,
    >
where
    Op: Optimizer<SId, Obj, Opt, Cod, Out, Scp>,
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
            FidThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
        > + Send
        + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Fidelity<Out> + Send + Sync,
    FnState: FuncState + Send + Sync,
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
                FidThrEvaluator::new(sobj.clone(), sopt.clone(), info)
            }
        };

        let (mut sobj, mut sopt, mut info): (_, _, Arc<Op::Info>);
        'main: loop {
            {
                let mut st = st.lock().unwrap();
                self.saver
                    .save_state(sp.clone(), self.optimizer.get_state(), &st, &eval);
                st.update(ExpStep::Evaluation);
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            let ((cobj, copt), cout) =
                <FidThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState> as ThrEvaluate<
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    Op::Info,
                    Op::SInfo,
                    SId,
                    Stepped<Obj, Cod, Out, FnState>,
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
                break 'main;
            };
            (sobj, sopt, info) = self.optimizer.step((cobj, copt), sp.clone());
            <FidThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState> as ThrEvaluate<
                St,
                Obj,
                Opt,
                Out,
                Cod,
                Op::Info,
                Op::SInfo,
                SId,
                Stepped<Obj, Cod, Out, FnState>,
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

    fn load(searchspace: Scp, objective: Stepped<Obj, Cod, Out, FnState>, mut saver: Sv) -> Self {
        let (stop, optimizer, evaluator) = saver
            .load(&searchspace, objective.get_codomain().as_ref())
            .unwrap();
        Saver::after_load(&mut saver, &searchspace, objective.get_codomain().as_ref());
        FidExperiment {
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
