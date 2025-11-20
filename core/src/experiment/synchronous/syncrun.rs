use crate::{
    checkpointer::Checkpointer,
    domain::onto::OntoDom,
    experiment::{
        BatchEvaluator, FidBatchEvaluator, FidThrBatchEvaluator, MonoEvaluate, Runable,
        ThrBatchEvaluator, ThrEvaluate,
    },
    objective::{outcome::FuncState, Codomain, FuncWrapper, Objective, Outcome},
    optimizer::Optimizer,
    recorder::Recorder,
    searchspace::Searchspace,
    solution::{partial::FidelityPartial, Batch, SId},
    stop::{ExpStep, Stop},
    FidOutcome, Id, Partial, Stepped,
};

use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::{DistCheckpointer, WorkerCheckpointer},
    experiment::{
        DistEvaluate, DistRunable, MasterWorker,
        mpi::{
            utils::{MPIProcess,checkpoint_order, stop_order},
            worker::{BaseWorker, FidWorker},
        }, synchronous::fidevaluator::FidDistBatchEvaluator,
    },
    recorder::DistRecorder,
};

//--------------------//
//--- MONOTHREADED ---//
//--------------------//

pub struct MonoExperiment<SolId, Eval, Scp, Op, St, Rec, Check, Obj, Opt, Out, Fn>
where
    SolId: Id,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Eval: MonoEvaluate<
        SolId,
        Obj,
        Opt,
        Op::SInfo,
        Op::Info,
        Op::Sol,
        St,
        Op::Cod,
        Out,
        Scp,
        Fn,
        Op::BType,
    >,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn>,
    St: Stop,
    Rec: Recorder<SolId, Obj, Opt, Out, Scp, Op, Fn, Op::BType>,
    Check: Checkpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Fn: FuncWrapper,
{
    pub searchspace: Scp,
    pub codomain: Op::Cod,
    pub objective: Fn,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    evaluator: Option<Eval>,
}

type OpBatch<Op, Obj, Opt, Out, Scp> = Batch<
    <Op as Optimizer<SId, Obj, Opt, Out, Scp, Objective<Obj, Out>>>::Sol,
    SId,
    Obj,
    Opt,
    <Op as Optimizer<SId, Obj, Opt, Out, Scp, Objective<Obj, Out>>>::SInfo,
    <Op as Optimizer<SId, Obj, Opt, Out, Scp, Objective<Obj, Out>>>::Info,
>;

type FOpBatch<Op, Obj, Opt, Out, Scp, FnState> = Batch<
    <Op as Optimizer<SId, Obj, Opt, Out, Scp, Stepped<Obj, Out, FnState>>>::Sol,
    SId,
    Obj,
    Opt,
    <Op as Optimizer<SId, Obj, Opt, Out, Scp, Stepped<Obj, Out, FnState>>>::SInfo,
    <Op as Optimizer<SId, Obj, Opt, Out, Scp, Stepped<Obj, Out, FnState>>>::Info,
>;

impl<Scp, Op, St, Rec, Check, Obj, Opt, Out>
    Runable<SId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Objective<Obj, Out>>
    for MonoExperiment<
        SId,
        BatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info>,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Obj,
        Opt,
        Out,
        Objective<Obj, Out>,
    >
where
    Op: Optimizer<
        SId,
        Obj,
        Opt,
        Out,
        Scp,
        Objective<Obj, Out>,
        BType = OpBatch<Op, Obj, Opt, Out, Scp>,
    >,
    Scp: Searchspace<Op::Sol, SId, Obj, Opt, Op::SInfo>,
    St: Stop,
    Rec: Recorder<SId, Obj, Opt, Out, Scp, Op, Objective<Obj, Out>, Op::BType>,
    Check: Checkpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Objective<Obj, Out>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> Self {
        let (recorder, checkpointer) = saver;
        let (searchspace, codomain) = space;
        let recorder = match recorder {
            Some(mut r) => {
                r.init(&searchspace, &codomain);
                Some(r)
            }
            None => None,
        };
        let checkpointer = match checkpointer {
            Some(mut c) => {
                c.init();
                Some(c)
            }
            None => None,
        };
        MonoExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            evaluator: None,
        }
    }

    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => BatchEvaluator::new(self.optimizer.first_step(&self.searchspace)),
        };

        let mut batch;
        let mut batch_raw;
        let mut batch_comp;
        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval)
            }
            if self.stop.stop() {
                break 'main;
            };

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) = MonoEvaluate::<
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                Op::Sol,
                St,
                Op::Cod,
                Out,
                Scp,
                Objective<Obj, Out>,
                Op::BType,
            >::evaluate(
                &mut eval, &self.objective, &self.codomain, &mut self.stop
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save(&batch_comp, &batch_raw, &self.searchspace, &self.codomain);
            }

            if self.stop.stop() {
                if let Some(c) = &self.checkpointer {
                    c.save_state(self.optimizer.get_state(), &self.stop, &eval)
                }
                break 'main;
            };
            batch = self.optimizer.step(batch_comp, &self.searchspace);
            MonoEvaluate::<
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                Op::Sol,
                St,
                Op::Cod,
                Out,
                Scp,
                Objective<Obj, Out>,
                Op::BType,
            >::update(&mut eval, batch);
            self.stop.update(ExpStep::Optimization);
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Objective<Obj, Out>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        let (state, stop, evaluator) = checkpointer.load().unwrap();
        let optimizer = Op::from_state(state);
        let recorder = match recorder {
            Some(mut rec) => {
                rec.after_load(&searchspace, &codomain);
                Some(rec)
            }
            None => None,
        };
        checkpointer.after_load();

        MonoExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
        }
    }
}

impl<Scp, Op, St, Rec, Check, Obj, Opt, Out, FnState>
    Runable<SId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Stepped<Obj, Out, FnState>>
    for MonoExperiment<
        SId,
        FidBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info, FnState>,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Obj,
        Opt,
        Out,
        Stepped<Obj, Out, FnState>,
    >
where
    Op: Optimizer<
        SId,
        Obj,
        Opt,
        Out,
        Scp,
        Stepped<Obj, Out, FnState>,
        BType = FOpBatch<Op, Obj, Opt, Out, Scp, FnState>,
    >,
    Scp: Searchspace<Op::Sol, SId, Obj, Opt, Op::SInfo>,
    St: Stop,
    Rec: Recorder<SId, Obj, Opt, Out, Scp, Op, Stepped<Obj, Out, FnState>, Op::BType>,
    Check: Checkpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: FidOutcome,
    Op::Sol: FidelityPartial<SId, Obj, Op::SInfo>,
    <Op::Sol as Partial<SId, Obj, Op::SInfo>>::Twin<Opt>: FidelityPartial<SId, Opt, Op::SInfo>,
    FnState: FuncState,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Stepped<Obj, Out, FnState>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> Self {
        let (recorder, checkpointer) = saver;
        let (searchspace, codomain) = space;
        let recorder = match recorder {
            Some(mut r) => {
                r.init(&searchspace, &codomain);
                Some(r)
            }
            None => None,
        };
        let checkpointer = match checkpointer {
            Some(mut c) => {
                c.init();
                Some(c)
            }
            None => None,
        };
        MonoExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            evaluator: None,
        }
    }

    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => FidBatchEvaluator::new(self.optimizer.first_step(&self.searchspace)),
        };

        let mut batch;
        let mut batch_raw;
        let mut batch_comp;
        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval);
            }
            if self.stop.stop() {
                break 'main;
            };

            // Evaluate batch
            (batch_raw, batch_comp) = MonoEvaluate::<
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                Op::Sol,
                St,
                Op::Cod,
                Out,
                Scp,
                Stepped<Obj, Out, FnState>,
                Op::BType,
            >::evaluate(
                &mut eval, &self.objective, &self.codomain, &mut self.stop
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save(&batch_comp, &batch_raw, &self.searchspace, &self.codomain);
            }

            if self.stop.stop() {
                if let Some(c) = &self.checkpointer {
                    c.save_state(self.optimizer.get_state(), &self.stop, &eval)
                }
                break 'main;
            };

            batch = self.optimizer.step(batch_comp, &self.searchspace);
            MonoEvaluate::<
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                Op::Sol,
                St,
                Op::Cod,
                Out,
                Scp,
                Stepped<Obj, Out, FnState>,
                Op::BType,
            >::update(&mut eval, batch);
            self.stop.update(ExpStep::Optimization);
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Stepped<Obj, Out, FnState>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        let (state, stop, evaluator) = checkpointer.load().unwrap();
        let optimizer = Op::from_state(state);
        let recorder = match recorder {
            Some(mut r) => {
                r.after_load(&searchspace, &codomain);
                Some(r)
            }
            None => None,
        };
        checkpointer.after_load();

        MonoExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
        }
    }
}

//---------------------//
//--- MULTITHREADED ---//
//---------------------//

pub struct ThrExperiment<SolId, Eval, Scp, Op, St, Rec, Check, Obj, Opt, Out, Fn>
where
    SolId: Id,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Eval: ThrEvaluate<
        SolId,
        Obj,
        Opt,
        Op::SInfo,
        Op::Info,
        Op::Sol,
        St,
        Op::Cod,
        Out,
        Scp,
        Fn,
        Op::BType,
    >,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn>,
    St: Stop,
    Rec: Recorder<SolId, Obj, Opt, Out, Scp, Op, Fn, Op::BType>,
    Check: Checkpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Fn: FuncWrapper,
{
    pub searchspace: Scp,
    pub codomain: Op::Cod,
    pub objective: Fn,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    evaluator: Option<Eval>,
}

impl<Scp, Op, St, Rec, Check, Obj, Opt, Out>
    Runable<SId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Objective<Obj, Out>>
    for ThrExperiment<
        SId,
        ThrBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info>,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Obj,
        Opt,
        Out,
        Objective<Obj, Out>,
    >
where
    Op: Optimizer<
        SId,
        Obj,
        Opt,
        Out,
        Scp,
        Objective<Obj, Out>,
        BType = OpBatch<Op, Obj, Opt, Out, Scp>,
    >,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::Sol: Partial<SId, Obj, Op::SInfo> + Send + Sync,
    <Op::Sol as Partial<SId, Obj, Op::SInfo>>::Twin<Opt>:
        Partial<SId, Opt, Op::SInfo, Twin<Obj> = Op::Sol> + Send + Sync,
    St: Stop + Send + Sync,
    Op::Cod: Codomain<Out> + Send + Sync,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Out: Outcome + Send + Sync,
    Scp: Searchspace<Op::Sol, SId, Obj, Opt, Op::SInfo> + Send + Sync,
    Rec: Recorder<SId, Obj, Opt, Out, Scp, Op, Objective<Obj, Out>, Op::BType> + Send + Sync,
    Check: Checkpointer,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Objective<Obj, Out>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, checkpointer) = saver;
        let recorder = match recorder {
            Some(mut r) => {
                r.init(&searchspace, &codomain);
                Some(r)
            }
            None => None,
        };
        let checkpointer = match checkpointer {
            Some(mut c) => {
                c.init();
                Some(c)
            }
            None => None,
        };
        ThrExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            evaluator: None,
        }
    }

    fn run(mut self) {
        let ob = Arc::new(self.objective);
        let cod = Arc::new(self.codomain);
        let st = Arc::new(Mutex::new(self.stop));
        let scp = Arc::new(self.searchspace);

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => ThrBatchEvaluator::new(self.optimizer.first_step(&scp)),
        };

        let mut batch;
        let mut batch_raw;
        let mut batch_comp;
        'main: loop {
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Iteration);
                if let Some(c) = &self.checkpointer {
                    c.save_state(self.optimizer.get_state(), &*st, &eval)
                }
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) =
                ThrEvaluate::<
                    SId,
                    Obj,
                    Opt,
                    Op::SInfo,
                    Op::Info,
                    Op::Sol,
                    St,
                    Op::Cod,
                    Out,
                    Scp,
                    Objective<Obj, Out>,
                    Op::BType,
                >::evaluate(&mut eval, ob.clone(), cod.clone(), st.clone());

            // Saver part
            if let Some(r) = &self.recorder {
                r.save(&batch_comp, &batch_raw, &scp, &cod);
            }

            {
                let st = st.lock().unwrap();
                if st.stop() {
                    if let Some(c) = &self.checkpointer {
                        c.save_state(self.optimizer.get_state(), &*st, &eval)
                    }
                    break 'main;
                };
            }
            batch = self.optimizer.step(batch_comp, &scp);
            ThrEvaluate::<
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                Op::Sol,
                St,
                Op::Cod,
                Out,
                Scp,
                Objective<Obj, Out>,
                Op::BType,
            >::update(&mut eval, batch);
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Optimization);
                if st.stop() {
                    if let Some(c) = &self.checkpointer {
                        c.save_state(self.optimizer.get_state(), &*st, &eval)
                    }
                    break 'main;
                };
            }
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Objective<Obj, Out>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        let (state, stop, evaluator) = checkpointer.load().unwrap();
        let optimizer = Op::from_state(state);
        let recorder = match recorder {
            Some(mut rec) => {
                rec.after_load(&searchspace, &codomain);
                Some(rec)
            }
            None => None,
        };
        checkpointer.after_load();
        ThrExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
        }
    }
}

impl<Scp, Op, St, Rec, Check, Obj, Opt, Out, FnState>
    Runable<SId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Stepped<Obj, Out, FnState>>
    for ThrExperiment<
        SId,
        FidThrBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info, FnState>,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Obj,
        Opt,
        Out,
        Stepped<Obj, Out, FnState>,
    >
where
    Op: Optimizer<
        SId,
        Obj,
        Opt,
        Out,
        Scp,
        Stepped<Obj, Out, FnState>,
        BType = FOpBatch<Op, Obj, Opt, Out, Scp, FnState>,
    >,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    St: Stop + Send + Sync,
    Out: FidOutcome + Send + Sync,
    Scp: Searchspace<Op::Sol, SId, Obj, Opt, Op::SInfo> + Send + Sync,
    Rec: Recorder<SId, Obj, Opt, Out, Scp, Op, Stepped<Obj, Out, FnState>, Op::BType> + Send + Sync,
    Check: Checkpointer,
    FnState: FuncState + Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::Sol: FidelityPartial<SId, Obj, Op::SInfo> + Send + Sync,
    <Op::Sol as Partial<SId, Obj, Op::SInfo>>::Twin<Opt>:
        FidelityPartial<SId, Opt, Op::SInfo, Twin<Obj> = Op::Sol> + Send + Sync,
    Op::Cod: Codomain<Out> + Send + Sync,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Stepped<Obj, Out, FnState>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> Self {
        let (recorder, checkpointer) = saver;
        let (searchspace, codomain) = space;
        let recorder = match recorder {
            Some(mut r) => {
                r.init(&searchspace, &codomain);
                Some(r)
            }
            None => None,
        };
        let checkpointer = match checkpointer {
            Some(mut c) => {
                c.init();
                Some(c)
            }
            None => None,
        };
        ThrExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            evaluator: None,
        }
    }

    fn run(mut self) {
        let ob = Arc::new(self.objective);
        let cod = Arc::new(self.codomain);
        let st = Arc::new(Mutex::new(self.stop));
        let scp = Arc::new(self.searchspace);

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => FidThrBatchEvaluator::new(self.optimizer.first_step(&scp)),
        };

        let mut batch;
        let mut batch_raw;
        let mut batch_comp;
        'main: loop {
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Iteration);
                if let Some(c) = &self.checkpointer {
                    c.save_state(self.optimizer.get_state(), &*st, &eval)
                }
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) =
                ThrEvaluate::<
                    SId,
                    Obj,
                    Opt,
                    Op::SInfo,
                    Op::Info,
                    Op::Sol,
                    St,
                    Op::Cod,
                    Out,
                    Scp,
                    Stepped<Obj, Out, FnState>,
                    Op::BType,
                >::evaluate(&mut eval, ob.clone(), cod.clone(), st.clone());

            // Saver part
            if let Some(r) = &self.recorder {
                r.save(&batch_comp, &batch_raw, &scp, &cod);
            }

            {
                let st = st.lock().unwrap();
                if st.stop() {
                    if let Some(c) = &self.checkpointer {
                        c.save_state(self.optimizer.get_state(), &*st, &eval)
                    }
                    break 'main;
                };
            }
            batch = self.optimizer.step(batch_comp, &scp);
            ThrEvaluate::<
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                Op::Sol,
                St,
                Op::Cod,
                Out,
                Scp,
                Stepped<Obj, Out, FnState>,
                Op::BType,
            >::update(&mut eval, batch);
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Optimization);
                if st.stop() {
                    if let Some(c) = &self.checkpointer {
                        c.save_state(self.optimizer.get_state(), &*st, &eval)
                    }
                    break 'main;
                };
            }
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Stepped<Obj, Out, FnState>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        let (state, stop, evaluator) = checkpointer.load().unwrap();
        let optimizer = Op::from_state(state);
        let recorder = match recorder {
            Some(mut rec) => {
                rec.after_load(&searchspace, &codomain);
                Some(rec)
            }
            None => None,
        };
        checkpointer.after_load();
        ThrExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
        }
    }
}

//-------------------//
//--- DISTRIBUTED ---//
//-------------------//

#[cfg(feature = "mpi")]
pub struct DistExperiment<'a, SolId, Eval, Scp, Op, St, Rec, Check, Obj, Opt, Out, Fn>
where
    SolId: Id,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Eval: DistEvaluate<
        SolId,
        Obj,
        Opt,
        Op::SInfo,
        Op::Info,
        Op::Sol,
        St,
        Op::Cod,
        Out,
        Scp,
        Fn,
        Op::BType,
    >,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn>,
    St: Stop,
    Rec: DistRecorder<SolId, Obj, Opt, Out, Scp, Op, Fn, Op::BType>,
    Check: DistCheckpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Fn: FuncWrapper,
{
    pub proc: &'a MPIProcess,
    pub searchspace: Scp,
    pub codomain: Op::Cod,
    pub objective: Fn,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    evaluator: Option<Eval>,
}

#[cfg(feature = "mpi")]
impl<'a, Scp, Op, St, Rec, Check, Obj, Opt, Out>
    DistRunable<'a, SId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Objective<Obj, Out>>
    for DistExperiment<
        'a,
        SId,
        BatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info>,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Obj,
        Opt,
        Out,
        Objective<Obj, Out>,
    >
where
    Self: 'a,
    Op: Optimizer<
        SId,
        Obj,
        Opt,
        Out,
        Scp,
        Objective<Obj, Out>,
        BType = OpBatch<Op, Obj, Opt, Out, Scp>,
    >,
    Scp: Searchspace<Op::Sol, SId, Obj, Opt, Op::SInfo>,
    St: Stop,
    Rec: DistRecorder<SId, Obj, Opt, Out, Scp, Op, Objective<Obj, Out>, Op::BType>,
    Check: DistCheckpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
{
    type WType = BaseWorker<'a, Obj, Out>;
    fn new(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Objective<Obj, Out>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> MasterWorker<'a, Self, SId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Objective<Obj, Out>>
    {
        let (searchspace, codomain) = space;
        let (recorder, checkpointer) = saver;
        if proc.rank == 0 {
            let recorder = match recorder {
                Some(mut r) => {
                    r.init_dist(proc, &searchspace, &codomain);
                    Some(r)
                }
                None => None,
            };
            let checkpointer = match checkpointer {
                Some(mut c) => {
                    c.init_dist(proc);
                    Some(c)
                }
                None => None,
            };
            MasterWorker::Master(DistExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer,
                evaluator: None,
            })
        } else {
            <Check as DistCheckpointer>::no_check_init(proc);
            MasterWorker::Worker(BaseWorker::new(objective, proc))
        }
    }

    fn load(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Objective<Obj, Out>,
        saver: (Option<Rec>, Check),
    ) -> MasterWorker<'a, Self, SId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Objective<Obj, Out>>
    {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        if proc.rank == 0 {
            let (state, stop, evaluator) = checkpointer.load_dist(proc.rank).unwrap();
            let optimizer = Op::from_state(state);
            let recorder = match recorder {
                Some(mut r) => {
                    r.after_load_dist(proc, &searchspace, &codomain);
                    Some(r)
                }
                None => None,
            };
            checkpointer.after_load_dist(proc);
            MasterWorker::Master(DistExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer: Some(checkpointer),
                evaluator: Some(evaluator),
            })
        } else {
            <Check as DistCheckpointer>::no_check_init(proc);
            MasterWorker::Worker(BaseWorker::new(objective, proc))
        }
    }
    
    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let batch = self.optimizer.first_step(&self.searchspace);
                BatchEvaluator::new(batch)
            }
        };

        let mut batch;
        let mut batch_raw;
        let mut batch_comp;

        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(
                    self.optimizer.get_state(),
                    &self.stop,
                    &eval,
                    self.proc.rank,
                );
            }
            if self.stop.stop() {
                break 'main;
            };
            
            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) = DistEvaluate::<
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                Op::Sol,
                St,
                Op::Cod,
                Out,
                Scp,
                Objective<Obj, Out>,
                Op::BType,
            >::evaluate(
                &mut eval,
                self.proc,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_dist(&batch_comp, &batch_raw, &self.searchspace, &self.codomain);
            }

            if self.stop.stop() {
                if let Some(c) = &self.checkpointer {
                    c.save_state_dist(
                        self.optimizer.get_state(),
                        &self.stop,
                        &eval,
                        self.proc.rank,
                    );
                }
                break 'main;
            };

            batch = self.optimizer.step(batch_comp, &self.searchspace);
            DistEvaluate::<
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                Op::Sol,
                St,
                Op::Cod,
                Out,
                Scp,
                Objective<Obj, Out>,
                Op::BType,
            >::update(&mut eval, batch);

            self.stop.update(ExpStep::Optimization);
        }
        stop_order(self.proc, 1..self.proc.size);
    }
}

#[cfg(feature = "mpi")]
impl<'a, Scp, Op, St, Rec, Check, Obj, Opt, Out, FnState>
    DistRunable<'a, SId, Scp, Op, St, Rec, Check, Out, Obj, Opt, Stepped<Obj, Out, FnState>>
    for DistExperiment<
        'a,
        SId,
        FidDistBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info>,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Obj,
        Opt,
        Out,
        Stepped<Obj, Out, FnState>,
    >
where
    Self: 'a,
    Op: Optimizer<
        SId,
        Obj,
        Opt,
        Out,
        Scp,
        Stepped<Obj, Out, FnState>,
        BType = FOpBatch<Op, Obj, Opt, Out, Scp, FnState>,
    >,
    Scp: Searchspace<Op::Sol, SId, Obj, Opt, Op::SInfo>,
    St: Stop,
    Rec: DistRecorder<SId, Obj, Opt, Out, Scp, Op, Stepped<Obj, Out, FnState>, Op::BType>,
    Check: DistCheckpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: FidOutcome,
    Op::Sol: FidelityPartial<SId, Obj, Op::SInfo>,
    <Op::Sol as Partial<SId, Obj, Op::SInfo>>::Twin<Opt>: FidelityPartial<SId, Opt, Op::SInfo>,
    FnState: FuncState,
{
    type WType = FidWorker<'a, SId, Obj, Out, FnState, Check>;
    fn new(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Stepped<Obj, Out, FnState>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> MasterWorker<
        'a,
        Self,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Obj,
        Opt,
        Stepped<Obj, Out, FnState>,
    > {
        let (recorder, checkpointer) = saver;
        let (searchspace, codomain) = space;
        if proc.rank == 0 {
            let recorder = match recorder {
                Some(mut r) => {
                    r.init_dist(proc, &searchspace, &codomain);
                    Some(r)
                }
                None => None,
            };
            let checkpointer = match checkpointer {
                Some(mut c) => {
                    c.init_dist(proc);
                    Some(c)
                }
                None => None,
            };
            MasterWorker::Master(DistExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer,
                evaluator: None,
            })
        } else {
            let check = match checkpointer {
                Some(c) => {
                    let mut wc = c.get_check_worker(proc);
                    println!("CHIOTE BITTE");
                    wc.init(proc);
                    Some(wc)
                }
                None => None,
            };
            MasterWorker::Worker(FidWorker::new(objective, check, proc))
        }
    }

    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => FidDistBatchEvaluator::new(self.optimizer.first_step(&self.searchspace)),
        };

        let mut batch;
        let mut batch_raw;
        let mut batch_comp;

        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(
                    self.optimizer.get_state(),
                    &self.stop,
                    &eval,
                    self.proc.rank,
                );
                checkpoint_order(self.proc, 1..self.proc.size);
            }
            if self.stop.stop() {
                break 'main;
            };

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) = DistEvaluate::<
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                Op::Sol,
                St,
                Op::Cod,
                Out,
                Scp,
                Stepped<Obj, Out, FnState>,
                Op::BType,
            >::evaluate(
                &mut eval,
                self.proc,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_dist(&batch_comp, &batch_raw, &self.searchspace, &self.codomain);
            }

            if self.stop.stop() {
                if let Some(c) = &self.checkpointer {
                    c.save_state_dist(
                        self.optimizer.get_state(),
                        &self.stop,
                        &eval,
                        self.proc.rank,
                    );
                    checkpoint_order(self.proc, 1..self.proc.size);
                }
                break 'main;
            };

            batch = self.optimizer.step(batch_comp, &self.searchspace);
            DistEvaluate::<
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                Op::Sol,
                St,
                Op::Cod,
                Out,
                Scp,
                Stepped<Obj, Out, FnState>,
                Op::BType,
            >::update(&mut eval, batch);

            self.stop.update(ExpStep::Optimization);
        }
        stop_order(self.proc, 1..self.proc.size);
    }

    fn load(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Stepped<Obj, Out, FnState>,
        saver: (Option<Rec>, Check),
    ) -> MasterWorker<
        'a,
        Self,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Obj,
        Opt,
        Stepped<Obj, Out, FnState>,
    > {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        if proc.rank == 0 {
            let (state, stop, evaluator) = checkpointer.load_dist(proc.rank).unwrap();
            let optimizer = Op::from_state(state);
            let recorder = match recorder {
                Some(mut r) => {
                    r.after_load_dist(proc, &searchspace, &codomain);
                    Some(r)
                }
                None => None,
            };
            checkpointer.after_load_dist(proc);
            MasterWorker::Master(DistExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer: Some(checkpointer),
                evaluator: Some(evaluator),
            })
        } else {
            let mut check = checkpointer.get_check_worker(proc);
            check.after_load(proc);
            MasterWorker::Worker(FidWorker::new(objective, Some(check), proc))
        }
    }
}