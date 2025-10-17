use crate::{
    Codomain, Cost, Partial, domain::Domain, experiment::{Evaluate, FidEvaluator, FidThrEvaluator, MonoEvaluate, Runable, ThrEvaluate}, objective::{Outcome, Stepped, outcome::FuncState}, optimizer::{CBType, OBType, opt::{OpCodType, OpInfType, OpSInfType, OpSolType, Optimizer}}, saver::Saver, searchspace::Searchspace, solution::{Batch, SId}, stop::{ExpStep, Stop}
};

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};
pub struct FidExperiment<Eval, Scp, Op, St, Sv, Obj, Opt, Out, FnState>
where
    Eval: Evaluate,
    Op: Optimizer<SId,Obj,Opt,Out,Scp,FnWrap = Stepped<Obj, OpCodType<Op,SId,Obj,Opt,Out,Scp>, Out, FnState>>,
    St: Stop,
    Scp: Searchspace<Op::Sol,SId,Obj,Opt,Op::SInfo>,
    Sv: Saver<SId, St, Obj, Opt, Out, Scp, Op, Eval>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    FnState: FuncState,
{
    pub searchspace: Scp,
    pub objective: Op::FnWrap,
    pub optimizer: Op,
    pub stop: St,
    pub saver: Sv,
    evaluator: Option<Eval>,
    _domobj: PhantomData<Obj>,
    _domopt: PhantomData<Opt>,
    _out: PhantomData<Out>,
}

impl<Scp, Op, St, Sv, Obj, Opt, Out, FnState>
    Runable<
        FidEvaluator<Op::Sol,SId,Obj,Opt,Op::SInfo,Op::Info, FnState>,
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
        FidEvaluator<Op::Sol,SId,Obj,Opt,Op::SInfo,Op::Info, FnState>,
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
    Op: Optimizer<SId, Obj, Opt, Out, Scp,
            FnWrap = Stepped<Obj,OpCodType<Op,SId,Obj,Opt,Out,Scp>,Out,FnState>,
            BType = Batch<
                        OpSolType<Op,SId,Obj,Opt,Out,Scp>,
                        SId,Obj,Opt,
                        OpSInfType<Op,SId,Obj,Opt,Out,Scp>,
                        OpInfType<Op,SId,Obj,Opt,Out,Scp>
                    >
    >,
    Scp: Searchspace<Op::Sol,SId,Obj,Opt,Op::SInfo>,
    St: Stop,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Out,
        Scp,
        Op,
        FidEvaluator<Op::Sol,SId,Obj,Opt,Op::SInfo,Op::Info, FnState>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    FnState:FuncState,
    Op::Cod: Cost<Out>,
{
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

        let mut batch: Op::BType;
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
            (batch_raw,batch_comp) = <
                FidEvaluator<Op::Sol,SId,Obj,Opt,Op::SInfo,Op::Info, FnState>
                as MonoEvaluate<Op,St,Obj,Opt,Out,SId,Scp>
            >::evaluate(&mut eval, ob.clone(), st.clone());

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
            
            <FidEvaluator<Op::Sol,SId,Obj,Opt,Op::SInfo,Op::Info, FnState>
                as MonoEvaluate<Op,St,Obj,Opt,Out,SId,Scp>
            >::update(&mut eval, batch);
            
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

impl<Scp, Op, St, Sv, Obj, Opt, Out, FnState>
    Runable<
        FidThrEvaluator<Op::Sol,SId,Obj,Opt,Op::SInfo,Op::Info, FnState>,
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
        FidThrEvaluator<Op::Sol,SId,Obj,Opt,Op::SInfo,Op::Info, FnState>,
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
    Op: Optimizer<SId, Obj, Opt, Out, Scp,
            FnWrap = Stepped<Obj,OpCodType<Op,SId,Obj,Opt,Out,Scp>,Out,FnState>,
            BType = Batch<
                        OpSolType<Op,SId,Obj,Opt,Out,Scp>,
                        SId,Obj,Opt,
                        OpSInfType<Op,SId,Obj,Opt,Out,Scp>,
                        OpInfType<Op,SId,Obj,Opt,Out,Scp>
                    >
    > + Send + Sync,
    Scp: Searchspace<Op::Sol,SId,Obj,Opt,Op::SInfo> + Send + Sync,
    St: Stop + Send + Sync,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Out,
        Scp,
        Op,
        FidThrEvaluator<Op::Sol,SId,Obj,Opt,Op::SInfo,Op::Info, FnState>,
    > + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Out: Outcome + Send + Sync,
    FnState:FuncState + Send + Sync,
    FnState: FuncState + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Op::Cod: Cost<Out> + Send + Sync,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::State: Send + Sync,
    Op::Sol: Send + Sync,
    <Op::Sol as Partial<SId,Obj,Op::SInfo>>::Twin<Opt>: Send + Sync,
{
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

        let mut batch: Op::BType;
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
            (batch_raw,batch_comp) =<
                FidThrEvaluator<Op::Sol,SId,Obj,Opt,Op::SInfo,Op::Info, FnState>
                as ThrEvaluate<Op,St,Obj,Opt,Out,SId,Scp>
            >::evaluate(&mut eval, ob.clone(), st.clone());

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
            
            <FidThrEvaluator<Op::Sol,SId,Obj,Opt,Op::SInfo,Op::Info, FnState>
                as ThrEvaluate<Op,St,Obj,Opt,Out,SId,Scp>
            >::update(&mut eval, batch);
            
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
