use crate::{
    checkpointer::Checkpointer,
    domain::onto::LinkOpt,
    experiment::{
        BatchEvaluator, FidBatchEvaluator, FidThrBatchEvaluator, MonoEvaluate, Runable,
        ThrBatchEvaluator, ThrEvaluate,
    },
    objective::{outcome::FuncState, FuncWrapper, Objective, Outcome},
    optimizer::opt::{BatchOptimizer, OpSInfType},
    recorder::Recorder,
    searchspace::{CompShape, Searchspace},
    solution::{shape::RawObj, HasFidelity, HasStep, SolutionShape, Uncomputed},
    stop::{ExpStep, Stop},
    FidOutcome, Id, Stepped,
};

#[cfg(feature = "mpi")]
use std::marker::PhantomData;
use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::{DistCheckpointer, WorkerCheckpointer},
    experiment::{
        mpi::{
            utils::{checkpoint_order, stop_order, FXMessage, MPIProcess, SendRec, XMessage, XMsg},
            worker::{BaseWorker, FidWorker},
        },
        batched::fidevaluator::FidDistBatchEvaluator,
        DistEvaluate, DistRunable, MasterWorker,
    },
    recorder::DistRecorder,
    solution::{
        shape::{SolObj, SolOpt},
        HasY,
    },
    SId,
};

//--------------------//
//--- MONOTHREADED ---//
//--------------------//

pub struct MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, Fn>,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SolId, Out, Scp, Op>,
    Check: Checkpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: MonoEvaluate<PSol, SolId, Op, Scp, Out, St, Fn>,
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

impl<PSol, SolId, Scp, Op, St, Rec, Check, Out>
    Runable<
        PSol,
        SolId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
    >
    for MonoExperiment<
        PSol,
        SolId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
        BatchEvaluator<SolId, Op::SInfo, Op::Info, Scp::SolShape>,
    >
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: BatchOptimizer<
        PSol,
        SolId,
        LinkOpt<Scp>,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>, Out>,
    >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SolId, Out, Scp, Op>,
    Check: Checkpointer,
    Out: Outcome,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
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
        let mut computed;
        let mut outputed;
        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval)
            }
            if self.stop.stop() {
                break 'main;
            };

            // Arc copy of data to send to evaluator thread.
            (computed, outputed) = MonoEvaluate::<_, SolId, Op, Scp, Out, St, _>::evaluate(
                &mut eval,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch(&computed, &outputed, &self.searchspace, &self.codomain);
            }

            if self.stop.stop() {
                if let Some(c) = &self.checkpointer {
                    c.save_state(self.optimizer.get_state(), &self.stop, &eval)
                }
                break 'main;
            };
            batch = self.optimizer.step(computed, &self.searchspace);
            MonoEvaluate::<_, SolId, Op, Scp, Out, St, _>::update(&mut eval, batch);
            self.stop.update(ExpStep::Optimization);
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
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

impl<PSol, SolId, Scp, Op, St, Rec, Check, Out, FnState>
    Runable<
        PSol,
        SolId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
    >
    for MonoExperiment<
        PSol,
        SolId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        FidBatchEvaluator<SolId, Op::SInfo, Op::Info, Scp::SolShape, FnState>,
    >
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: BatchOptimizer<
        PSol,
        SolId,
        LinkOpt<Scp>,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>, Out, FnState>,
    >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Rec: Recorder<PSol, SolId, Out, Scp, Op>,
    Check: Checkpointer,
    Out: FidOutcome,
    FnState: FuncState,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
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
        let mut computed;
        let mut outputed;
        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval);
            }
            if self.stop.stop() {
                break 'main;
            };

            // Evaluate batch
            (computed, outputed) = MonoEvaluate::<_, SolId, Op, Scp, Out, St, _>::evaluate(
                &mut eval,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch(&computed, &outputed, &self.searchspace, &self.codomain);
            }

            if self.stop.stop() {
                if let Some(c) = &self.checkpointer {
                    c.save_state(self.optimizer.get_state(), &self.stop, &eval)
                }
                break 'main;
            };

            batch = self.optimizer.step(computed, &self.searchspace);
            MonoEvaluate::<_, SolId, Op, Scp, Out, St, _>::update(&mut eval, batch);
            self.stop.update(ExpStep::Optimization);
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
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

pub struct ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, Fn>,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SolId, Out, Scp, Op>,
    Check: Checkpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: ThrEvaluate<PSol, SolId, Op, Scp, Out, St, Fn>,
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

impl<PSol, SolId, Scp, Op, St, Rec, Check, Out>
    Runable<
        PSol,
        SolId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
    >
    for ThrExperiment<
        PSol,
        SolId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
        ThrBatchEvaluator<SolId, Op::SInfo, Op::Info, Scp::SolShape>,
    >
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id + Send + Sync,
    Op: BatchOptimizer<
        PSol,
        SolId,
        LinkOpt<Scp>,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>, Out>,
    >,
    Op::Cod: Send + Sync,
    Op::Info: Send + Sync,
    Op::SInfo: Send + Sync,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    Scp::SolShape: Send + Sync,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        Debug + SolutionShape<SolId, Op::SInfo> + Send + Sync,
    St: Stop + Send + Sync,
    Out: Outcome + Send + Sync,
    Rec: Recorder<PSol, SolId, Out, Scp, Op>,
    Check: Checkpointer,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
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
        let mut computed;
        let mut outputed;
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
            (computed, outputed) = ThrEvaluate::<_, SolId, Op, Scp, Out, St, _>::evaluate(
                &mut eval,
                ob.clone(),
                cod.clone(),
                st.clone(),
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch(&computed, &outputed, &scp, &cod);
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
            batch = self.optimizer.step(computed, &scp);
            ThrEvaluate::<_, SolId, Op, Scp, Out, St, _>::update(&mut eval, batch);
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
        objective: Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
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

impl<PSol, SolId, Scp, Op, St, Rec, Check, Out, FnState>
    Runable<
        PSol,
        SolId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
    >
    for ThrExperiment<
        PSol,
        SolId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        FidThrBatchEvaluator<SolId, Op::SInfo, Op::Info, Scp::SolShape, FnState>,
    >
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id + Send + Sync,
    Op: BatchOptimizer<
        PSol,
        SolId,
        LinkOpt<Scp>,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>, Out, FnState>,
    >,
    Op::Cod: Send + Sync,
    Op::Info: Send + Sync,
    Op::SInfo: Send + Sync,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    Scp::SolShape: HasStep + HasFidelity + Send + Sync,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + Debug + Send + Sync,
    St: Stop + Send + Sync,
    Out: FidOutcome + Send + Sync,
    FnState: FuncState + Send + Sync,
    Rec: Recorder<PSol, SolId, Out, Scp, Op>,
    Check: Checkpointer,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
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
        let mut computed;
        let mut outputed;
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
            (computed, outputed) = ThrEvaluate::<_, SolId, Op, Scp, Out, St, _>::evaluate(
                &mut eval,
                ob.clone(),
                cod.clone(),
                st.clone(),
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch(&computed, &outputed, &scp, &cod);
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
            batch = self.optimizer.step(computed, &scp);
            ThrEvaluate::<_, SolId, Op, Scp, Out, St, _>::update(&mut eval, batch);
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
        objective: Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
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
pub struct DistExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval, M>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, Fn>,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    St: Stop,
    Rec: DistRecorder<PSol, SolId, Out, Scp, Op>,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: DistEvaluate<PSol, SolId, Op, Scp, Out, St, Fn, M>,
    M: XMsg<SolObj<Scp::SolShape, SolId, Op::SInfo>, SolId, Scp::Obj, Op::SInfo>,
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
    msg: PhantomData<M>,
}

#[cfg(feature = "mpi")]
impl<'a, PSol, Scp, Op, St, Rec, Check, Out>
    DistRunable<
        'a,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
    >
    for DistExperiment<
        'a,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        BatchEvaluator<SId, Op::SInfo, Op::Info, Scp::SolShape>,
        XMessage<SId, RawObj<Scp::SolShape, SId, Op::SInfo>>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    Op: BatchOptimizer<
        PSol,
        SId,
        LinkOpt<Scp>,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SId, OpSInfType<Op, PSol, Scp, SId, Out>>, Out>,
    >,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>>,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + HasY<Op::Cod, Out>,
    St: Stop,
    Rec: DistRecorder<PSol, SId, Out, Scp, Op>,
    Check: DistCheckpointer,
    Out: Outcome,
{
    type WType = BaseWorker<'a, RawObj<Scp::SolShape, SId, Op::SInfo>, Out>;
    fn new(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> MasterWorker<
        'a,
        Self,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
    > {
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
                msg: PhantomData,
            })
        } else {
            <Check as DistCheckpointer>::no_check_init(proc);
            MasterWorker::Worker(BaseWorker::new(objective, proc))
        }
    }

    fn load(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        saver: (Option<Rec>, Check),
    ) -> MasterWorker<
        'a,
        Self,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
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
                msg: PhantomData,
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
        // Define send/rec utilitaries and parameters
        let config = bincode::config::standard(); // Bytes encoding config
        let mut sendrec = SendRec::<
            '_,
            XMessage<SId, RawObj<Scp::SolShape, SId, Op::SInfo>>,
            Scp::SolShape,
            SId,
            Op::SInfo,
            Op::Cod,
            Out,
        >::new(config, self.proc);

        let mut batch;
        let mut computed;
        let mut outputed;

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
            (computed, outputed) = DistEvaluate::<_, SId, Op, Scp, Out, St, _, _>::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch_dist(&computed, &outputed, &self.searchspace, &self.codomain);
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

            batch = self.optimizer.step(computed, &self.searchspace);
            DistEvaluate::<_, SId, Op, Scp, Out, St, _, _>::update(&mut eval, batch);

            self.stop.update(ExpStep::Optimization);
        }
        stop_order(self.proc, 1..self.proc.size);
    }
}

#[cfg(feature = "mpi")]
impl<'a, PSol, Scp, Op, St, Rec, Check, Out, FnState>
    DistRunable<
        'a,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
    >
    for DistExperiment<
        'a,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        FidDistBatchEvaluator<SId, Op::SInfo, Op::Info, Scp::SolShape>,
        FXMessage<SId, RawObj<Scp::SolShape, SId, Op::SInfo>>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    Op: BatchOptimizer<
        PSol,
        SId,
        LinkOpt<Scp>,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, OpSInfType<Op, PSol, Scp, SId, Out>>, Out, FnState>,
    >,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    SolObj<Scp::SolShape, SId, Op::SInfo>: HasStep + HasFidelity,
    SolOpt<Scp::SolShape, SId, Op::SInfo>: HasStep + HasFidelity,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + HasY<Op::Cod, Out> + HasStep + HasFidelity,
    St: Stop,
    Rec: DistRecorder<PSol, SId, Out, Scp, Op>,
    Check: DistCheckpointer,
    Out: FidOutcome,
    FnState: FuncState,
{
    type WType = FidWorker<'a, SId, RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState, Check>;
    fn new(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> MasterWorker<
        'a,
        Self,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
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
                msg: PhantomData,
            })
        } else {
            let check = match checkpointer {
                Some(c) => {
                    let mut wc = c.get_check_worker(proc);
                    wc.init(proc);
                    Some(wc)
                }
                None => None,
            };
            MasterWorker::Worker(FidWorker::new(objective, check, proc))
        }
    }

    fn load(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        saver: (Option<Rec>, Check),
    ) -> MasterWorker<
        'a,
        Self,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
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
                msg: PhantomData,
            })
        } else {
            let mut check = checkpointer.get_check_worker(proc);
            let state = check.load(proc.rank).unwrap();
            check.after_load(proc);
            let mut w = FidWorker::new(objective, Some(check), proc);
            w.state = state;
            MasterWorker::Worker(w)
        }
    }

    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => FidDistBatchEvaluator::new(
                self.optimizer.first_step(&self.searchspace),
                self.proc.size as usize,
            ),
        };
        // Define send/rec utilitaries and parameters
        let config = bincode::config::standard(); // Bytes encoding config
        let mut sendrec = SendRec::<
            '_,
            FXMessage<SId, RawObj<Scp::SolShape, SId, Op::SInfo>>,
            Scp::SolShape,
            SId,
            Op::SInfo,
            Op::Cod,
            Out,
        >::new(config, self.proc);

        let mut batch;
        let mut computed;
        let mut outputed;

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
            (computed, outputed) = DistEvaluate::<_, SId, Op, Scp, Out, St, _, _>::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch_dist(&computed, &outputed, &self.searchspace, &self.codomain);
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

            batch = self.optimizer.step(computed, &self.searchspace);
            DistEvaluate::<_, SId, Op, Scp, Out, St, _, _>::update(&mut eval, batch);

            self.stop.update(ExpStep::Optimization);
        }
        stop_order(self.proc, 1..self.proc.size);
    }
}
