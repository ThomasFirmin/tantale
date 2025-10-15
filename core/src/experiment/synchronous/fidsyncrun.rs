use crate::{
    Codomain, Cost, domain::Domain, experiment::{Evaluate, FidEvaluator, FidThrEvaluator, MonoEvaluate, Runable, ThrEvaluate}, objective::{Outcome, Stepped, outcome::FuncState}, optimizer::{CBType, OBType, opt::{OpCodType, OpInfType, OpSInfType, Optimizer}}, saver::Saver, searchspace::Searchspace, solution::{Batch, SId}, stop::{ExpStep, Stop}
};

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};
pub struct FidExperiment<Eval, Scp, Op, St, Sv, Obj, Opt, Out, FnState>
where
    Eval: Evaluate,
    Op: Optimizer<SId, Obj, Opt, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: Saver<SId, St, Obj, Opt, Out, Scp, Op, Eval>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    FnState: FuncState,
{
    pub searchspace: Scp,
    pub objective: Stepped<Obj, Op::Cod, Out, FnState>,
    pub optimizer: Op,
    pub stop: St,
    pub saver: Sv,
    evaluator: Option<Eval>,
    _domobj: PhantomData<Obj>,
    _domopt: PhantomData<Opt>,
    _out: PhantomData<Out>,
}

impl<Scp, Op, St, Sv, Obj, Opt, Out,FnState>
    Runable<
        SId,
        Scp,
        Op,
        St,
        Sv,
        Out,
        Obj,
        Opt,
    >
    for FidExperiment<
        FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo,FnState>,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
        FnState,
    >
where
    Op:Optimizer<
        SId,Obj,Opt,Out,Scp,
        FnWrap = Stepped<Obj, OpCodType<Op,SId,Obj,Opt,Out,Scp>, Out,FnState>,
        BType = Batch<SId,Obj,Opt,OpSInfType<Op,SId,Obj,Opt,Out,Scp>,OpInfType<Op,SId,Obj,Opt,Out,Scp>>
    >,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo> + Send + Sync,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Out,
        Scp,
        Op,
        FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo,FnState>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    FnState:FuncState,
    Op::Cod: Cost<Out>,
{
    type Eval = FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo,FnState>;
    fn new(
        searchspace: Scp,
        objective: Op::FnWrap,
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
            _out: PhantomData,
        }
    }

    fn run(mut self) {
        let sp = Arc::new(self.searchspace);
        let ob = Arc::new(self.objective);
        let cod = ob.get_codomain();
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let batch = self.optimizer.first_step(sp.clone());
                FidEvaluator::new(batch)
            }
        };

        let mut batch: Batch<SId,Obj,Opt,Op::SInfo,Op::Info>;
        let mut batch_raw: OBType<Op,SId,Obj,Opt,Out,Scp>;
        let mut batch_comp: CBType<Op,SId,Obj,Opt,Out,Scp>;
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
            (batch_raw,batch_comp) = <Self::Eval as MonoEvaluate<Op,St,Obj,Opt,Out,SId,Scp>>::evaluate(&mut eval, ob.clone(), st.clone());

            // Saver part
            self.saver.save_partial(
                &eval.batch,
                sp.clone(),
            );
            self.saver.save_out(&batch_raw, sp.clone());
            self.saver.save_codom(&batch_comp, sp.clone(), cod.clone());

            if st.lock().unwrap().stop() {
                self.saver.save_state(
                    sp.clone(),
                    self.optimizer.get_state(),
                    &st.lock().unwrap(),
                    &eval,
                );
                break 'main;
            };
            batch = self.optimizer.step(batch_comp, sp.clone());
            <Self::Eval as MonoEvaluate<Op,St,Obj,Opt,Out,SId,Scp>>::update(&mut eval, batch);
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

    fn load(searchspace: Scp, objective: Op::FnWrap, mut saver: Sv) -> Self {
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
            _out: PhantomData,
        }
    }
}

impl<Scp, Op, St, Sv, Obj, Opt, Out,FnState>
    Runable<
        SId,
        Scp,
        Op,
        St,
        Sv,
        Out,
        Obj,
        Opt,
    >
    for FidExperiment<
        FidThrEvaluator<SId,Obj,Opt,Op::Info,Op::SInfo,FnState>,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
        FnState,
    >
where
    Op:Optimizer<
        SId,Obj,Opt,Out,Scp,
        FnWrap = Stepped<Obj, OpCodType<Op,SId,Obj,Opt,Out,Scp>, Out,FnState>,
        BType = Batch<SId,Obj,Opt,OpSInfType<Op,SId,Obj,Opt,Out,Scp>,OpInfType<Op,SId,Obj,Opt,Out,Scp>>
    >,
    St: Stop + Send + Sync,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo> + Send + Sync,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Out,
        Scp,
        Op,
        FidThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo,FnState>,
    > + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Out: Outcome + Send + Sync,
    FnState: FuncState + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Op::Cod: Cost<Out> + Send + Sync,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::State: Send + Sync,
{
    type Eval = FidThrEvaluator<SId,Obj,Opt,Op::Info,Op::SInfo,FnState>;
    fn new(
        searchspace: Scp,
        objective: Op::FnWrap,
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
            _out: PhantomData,
        }
    }
    
    fn run(mut self) {
        let sp = Arc::new(self.searchspace);
        let ob = Arc::new(self.objective);
        let cod = ob.get_codomain();
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let batch = self.optimizer.first_step(sp.clone());
                FidThrEvaluator::new(batch)
            }
        };

        let mut batch: Batch<SId,Obj,Opt,Op::SInfo,Op::Info>;
        let mut batch_raw: OBType<Op,SId,Obj,Opt,Out,Scp>;
        let mut batch_comp: CBType<Op,SId,Obj,Opt,Out,Scp>;
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
            (batch_raw,batch_comp) =<Self::Eval as ThrEvaluate<Op,St,Obj,Opt,Out,SId,Scp>>::evaluate(&mut eval, ob.clone(), st.clone());

            // Saver part
            rayon::join(
                || {
                    rayon::join(
                        ||
                        {
                            let _ = &self.saver.save_partial(&eval.batch, sp.clone());
                        }
                        , 
                        ||
                        {
                            let _ = &self.saver.save_info(&eval.batch, sp.clone());
                        }
                    );
                },
                || {
                    rayon::join(
                        ||
                        {
                            let _ = &self.saver.save_out(&batch_raw, sp.clone());
                        }
                        , 
                        ||
                        {
                            let _ = &self.saver.save_codom(&batch_comp, sp.clone(), cod.clone());
                        }
                    );
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
            batch = self.optimizer.step(batch_comp, sp.clone());
            <Self::Eval as ThrEvaluate<Op,St,Obj,Opt,Out,SId,Scp>>::update(&mut eval, batch);
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

    fn load(searchspace: Scp, objective: Op::FnWrap, mut saver: Sv) -> Self {
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
            _out: PhantomData,
        }
    }
}
