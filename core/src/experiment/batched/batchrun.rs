use crate::{
    FidOutcome, SId, Stepped, ThrCheckpointer,
    checkpointer::{FuncStateCheckpointer, MonoCheckpointer},
    domain::onto::LinkOpt,
    experiment::{
        BatchEvaluator, FidBatchEvaluator, FidThrBatchEvaluator, MonoEvaluate, MonoExperiment,
        OutBatchEvaluate, Runable, ThrBatchEvaluator, ThrEvaluate, ThrExperiment,
        basics::IdxMapPool,
    },
    objective::{Objective, Outcome, outcome::FuncState},
    optimizer::opt::{BatchOptimizer, OpSInfType},
    recorder::Recorder,
    searchspace::{CompShape, Searchspace},
    solution::{HasFidelity, HasStep, SolutionShape, Uncomputed, shape::RawObj},
    stop::{ExpStep, Stop},
};

use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::{DistCheckpointer, WorkerCheckpointer},
    experiment::{
        DistEvaluate, MPIExperiment, MPIRunable, MasterWorker,
        batched::batchfidevaluator::FidDistBatchEvaluator,
        mpi::{
            utils::{FXMessage, MPIProcess, SendRec, XMessage, stop_order},
            worker::{BaseWorker, FidWorker},
        },
    },
    recorder::DistRecorder,
    solution::{
        HasY,
        shape::{SolObj, SolOpt},
    },
};

//--------------------//
//--- MONOTHREADED ---//
//--------------------//
impl<PSol, Scp, Op, St, Rec, Check, Out>
    Runable<
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
    for MonoExperiment<
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
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>: SolutionShape<SId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: MonoCheckpointer,
    Out: Outcome,
{
    /// Create a new [`MonoExperiment`] from a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// [`Objective`], [`BatchOptimizer`], [`Stop`] condition and optional [`Recorder`] and [`Checkpointer`].
    fn new(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
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

    /// Load a [`MonoExperiment`] from a saved state using a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// and [`Objective`], along with an optional [`Recorder`] and non-optional [`Checkpointer`].
    /// You can use [`load!`](crate::load) macro to load an experiment more easily.
    fn load(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        checkpointer.before_load();

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

    /// Run the [`MonoExperiment`], performing optimization, , using a [`BatchOptimizer`], until the [`Stop`] condition is met.
    /// The process evaluates [`Batch`](crate::Batch)es of [`Uncomputed`] using the inner [`BatchEvaluator`],
    /// A checkpoint is performed after each optimization step. And [`CompBatch`](crate::Batch)es of [`Computed`](crate::Computed),
    /// are saved using the inner [`Recorder`] when [`BatchEvaluator`] has finished evaluating all elements.    saved using the inner [`Recorder`] when [`ThrBatchEvaluator`] has finished evaluating all elements.
    ///
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`BatchEvaluator`] updates) step.
    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => BatchEvaluator::new(self.optimizer.first_step(&self.searchspace)),
        };

        let mut batch;
        let mut computed;
        let mut outputed;

        self.stop.init();
        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if self.stop.stop() {
                break 'main;
            };

            // Evaluation part
            (computed, outputed) = MonoEvaluate::<
                _,
                SId,
                Op,
                Scp,
                Out,
                St,
                _,
                OutBatchEvaluate<SId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
            >::evaluate(
                &mut eval, &self.objective, &self.codomain, &mut self.stop
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch(&computed, &outputed, &self.searchspace, &self.codomain);
            }

            // Optimizer part
            batch = self.optimizer.step(computed, &self.searchspace);
            eval.update(batch);
            self.stop.update(ExpStep::Optimization);

            // Checkpointing part
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_mut_state(), &self.stop, &eval)
            }
        }
    }

    fn get_stop(&self) -> &St {
        &self.stop
    }

    fn get_searchspace(&self) -> &Scp {
        &self.searchspace
    }

    fn get_codomain(&self) -> &Op::Cod {
        &self.codomain
    }

    fn get_objective(&self) -> &Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out> {
        &self.objective
    }

    fn get_optimizer(&self) -> &Op {
        &self.optimizer
    }

    fn get_recorder(&self) -> Option<&Rec> {
        self.recorder.as_ref()
    }

    fn get_checkpointer(&self) -> Option<&Check> {
        self.checkpointer.as_ref()
    }

    fn get_mut_stop(&mut self) -> &mut St {
        &mut self.stop
    }

    fn get_mut_searchspace(&mut self) -> &mut Scp {
        &mut self.searchspace
    }

    fn get_mut_codomain(&mut self) -> &mut Op::Cod {
        &mut self.codomain
    }

    fn get_mut_objective(&mut self) -> &mut Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out> {
        &mut self.objective
    }

    fn get_mut_optimizer(&mut self) -> &mut Op {
        &mut self.optimizer
    }

    fn get_mut_recorder(&mut self) -> Option<&mut Rec> {
        self.recorder.as_mut()
    }

    fn get_mut_checkpointer(&mut self) -> Option<&mut Check> {
        self.checkpointer.as_mut()
    }
}

impl<PSol, Scp, Op, St, Rec, Check, Out, FnState>
    Runable<
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
    for MonoExperiment<
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        FidBatchEvaluator<
            SId,
            Op::SInfo,
            Op::Info,
            Scp::SolShape,
            FnState,
            IdxMapPool<SId, FnState, Check::FnStateCheck>,
        >,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity,
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
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: MonoCheckpointer,
    Out: FidOutcome,
    FnState: FuncState,
{
    /// Create a new [`MonoExperiment`] from a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// [`Stepped`], [`BatchOptimizer`], [`Stop`] condition and optional [`Recorder`] and [`Checkpointer`].
    fn new(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
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

    /// Load a [`MonoExperiment`] from a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// and [`Stepped`], along with an optional [`Recorder`] and non-optional [`Checkpointer`].
    /// You can use [`load!`](crate::load) macro to load an experiment more easily.
    /// All [`FuncState`]s saved in the checkpoint will be restored in the loaded experiment.
    fn load(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        checkpointer.before_load();
        let fn_check = checkpointer.new_func_state_checkpointer();

        let opt_state = checkpointer.load_optimizer().unwrap();
        let stop = checkpointer.load_stop().unwrap();
        let mut evaluator: FidBatchEvaluator<
            _,
            _,
            _,
            _,
            _,
            IdxMapPool<SId, FnState, Check::FnStateCheck>,
        > = checkpointer.load_evaluate().unwrap();
        let fn_states = fn_check.load_all_func_state();
        evaluator.pool = IdxMapPool::from_iter(fn_states);

        let optimizer = Op::from_state(opt_state);
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

    /// Run the [`MonoExperiment`], performing optimization, using a [`BatchOptimizer`], until the [`Stop`] condition is met.
    /// The process evaluates [`Batch`](crate::Batch)es of [`Uncomputed`] using the inner [`FidBatchEvaluator`],
    /// A checkpoint is performed after each optimization step. And [`CompBatch`](crate::Batch)es of [`Computed`](crate::Computed),
    /// are saved using the inner [`Recorder`] when [`FidBatchEvaluator`] has finished evaluating all elements.
    ///
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`FidBatchEvaluator`] updates) step.
    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let fn_check = self
                    .checkpointer
                    .as_ref()
                    .map(|c| c.new_func_state_checkpointer());
                FidBatchEvaluator::new(
                    self.optimizer.first_step(&self.searchspace),
                    IdxMapPool::new(fn_check),
                )
            }
        };

        let mut batch;
        let mut computed;
        let mut outputed;

        self.stop.init();
        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if self.stop.stop() {
                break 'main;
            };

            // Evaluation part
            (computed, outputed) = MonoEvaluate::<
                _,
                SId,
                Op,
                Scp,
                Out,
                St,
                _,
                OutBatchEvaluate<SId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
            >::evaluate(
                &mut eval, &self.objective, &self.codomain, &mut self.stop
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch(&computed, &outputed, &self.searchspace, &self.codomain);
            }

            // Optimizer part
            batch = self.optimizer.step(computed, &self.searchspace);
            eval.update(batch);
            self.stop.update(ExpStep::Optimization);

            // Checkpointing part
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_mut_state(), &self.stop, &eval);
            }
        }
    }

    fn get_stop(&self) -> &St {
        &self.stop
    }

    fn get_searchspace(&self) -> &Scp {
        &self.searchspace
    }

    fn get_codomain(&self) -> &Op::Cod {
        &self.codomain
    }

    fn get_objective(&self) -> &Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
        &self.objective
    }

    fn get_optimizer(&self) -> &Op {
        &self.optimizer
    }

    fn get_recorder(&self) -> Option<&Rec> {
        self.recorder.as_ref()
    }

    fn get_checkpointer(&self) -> Option<&Check> {
        self.checkpointer.as_ref()
    }

    fn get_mut_stop(&mut self) -> &mut St {
        &mut self.stop
    }

    fn get_mut_searchspace(&mut self) -> &mut Scp {
        &mut self.searchspace
    }

    fn get_mut_codomain(&mut self) -> &mut Op::Cod {
        &mut self.codomain
    }

    fn get_mut_objective(
        &mut self,
    ) -> &mut Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
        &mut self.objective
    }

    fn get_mut_optimizer(&mut self) -> &mut Op {
        &mut self.optimizer
    }

    fn get_mut_recorder(&mut self) -> Option<&mut Rec> {
        self.recorder.as_mut()
    }

    fn get_mut_checkpointer(&mut self) -> Option<&mut Check> {
        self.checkpointer.as_mut()
    }
}

//---------------------//
//--- MULTITHREADED ---//
//---------------------//

impl<PSol, Scp, Op, St, Rec, Check, Out>
    Runable<
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
    for ThrExperiment<
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        ThrBatchEvaluator<SId, Op::SInfo, Op::Info, Scp::SolShape>,
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
    Op::Cod: Send + Sync,
    Op::Info: Send + Sync,
    Op::SInfo: Send + Sync,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>>,
    Scp::SolShape: Send + Sync,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>: SolutionShape<SId, Op::SInfo> + Send + Sync,
    St: Stop + Send + Sync,
    Out: Outcome + Send + Sync,
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: ThrCheckpointer,
{
    /// Create a new [`ThrExperiment`] from a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// [`Objective`], [`BatchOptimizer`], [`Stop`] condition and optional [`Recorder`] and [`Checkpointer`].
    /// It also uses an internal [`ThrBatchEvaluator`] to evaluate a batch of [`Uncomputed`] in parallel.
    fn new(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
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
                c.init_thr();
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

    /// Load a [`ThrExperiment`] from a saved state using a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// and [`Objective`], along with an optional [`Recorder`] and non-optional [`Checkpointer`].
    /// You can use [`load!`](crate::load) macro to load an experiment more easily.
    fn load(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        checkpointer.before_load_thr();

        let opt_state = checkpointer.load_optimizer_thr().unwrap();
        let stop = checkpointer.load_stop_thr().unwrap();
        let evaluator = checkpointer.load_all_evaluate_thr().unwrap().pop().expect(
            "Only one evaluator state should be within the checkpoint for batched ThrExperiment",
        );

        let optimizer = Op::from_state(opt_state);
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

    /// Run the [`ThrExperiment`], performing optimization, using a [`BatchOptimizer`], until the [`Stop`] condition is met.
    /// The process evaluates [`Batch`](crate::Batch)es of [`Uncomputed`] using the inner [`ThrBatchEvaluator`],
    /// A checkpoint is performed after each optimization step. And [`CompBatch`](crate::Batch)es of [`Computed`](crate::Computed),
    /// are saved using the inner [`Recorder`] when [`ThrBatchEvaluator`] has finished evaluating all elements.
    ///
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`ThrBatchEvaluator`] updates) step.
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

        st.lock().unwrap().init();
        'main: loop {
            // Stop part
            {
                let mut stlock = st.lock().unwrap();
                if stlock.stop() {
                    break 'main;
                };
                stlock.update(ExpStep::Iteration);
            }

            // Evaluation part
            (computed, outputed) =
                ThrEvaluate::<
                    _,
                    SId,
                    Op,
                    Scp,
                    Out,
                    St,
                    _,
                    OutBatchEvaluate<SId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
                >::evaluate(&mut eval, ob.clone(), cod.clone(), st.clone());

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch(&computed, &outputed, &scp, &cod);
            }

            // Optimizer part
            batch = self.optimizer.step(computed, &scp);
            eval.update(batch);

            // Stop and checkpointing part
            {
                let mut stlock = st.lock().unwrap();
                stlock.update(ExpStep::Optimization);
                if let Some(c) = &self.checkpointer {
                    c.save_state_thr(self.optimizer.get_mut_state(), &*stlock, &eval, 0)
                }
            }
        }
    }

    fn get_stop(&self) -> &St {
        &self.stop
    }

    fn get_searchspace(&self) -> &Scp {
        &self.searchspace
    }

    fn get_codomain(&self) -> &Op::Cod {
        &self.codomain
    }

    fn get_objective(&self) -> &Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out> {
        &self.objective
    }

    fn get_optimizer(&self) -> &Op {
        &self.optimizer
    }

    fn get_recorder(&self) -> Option<&Rec> {
        self.recorder.as_ref()
    }

    fn get_checkpointer(&self) -> Option<&Check> {
        self.checkpointer.as_ref()
    }

    fn get_mut_stop(&mut self) -> &mut St {
        &mut self.stop
    }

    fn get_mut_searchspace(&mut self) -> &mut Scp {
        &mut self.searchspace
    }

    fn get_mut_codomain(&mut self) -> &mut Op::Cod {
        &mut self.codomain
    }

    fn get_mut_objective(&mut self) -> &mut Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out> {
        &mut self.objective
    }

    fn get_mut_optimizer(&mut self) -> &mut Op {
        &mut self.optimizer
    }

    fn get_mut_recorder(&mut self) -> Option<&mut Rec> {
        self.recorder.as_mut()
    }

    fn get_mut_checkpointer(&mut self) -> Option<&mut Check> {
        self.checkpointer.as_mut()
    }
}

impl<PSol, Scp, Op, St, Rec, Check, Out, FnState>
    Runable<
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
    for ThrExperiment<
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        FidThrBatchEvaluator<
            SId,
            Op::SInfo,
            Op::Info,
            Scp::SolShape,
            FnState,
            IdxMapPool<SId, FnState, Check::FnStateCheck>,
        >,
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
    Op::Cod: Send + Sync,
    Op::Info: Send + Sync,
    Op::SInfo: Send + Sync,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>>,
    Scp::SolShape: HasStep + HasFidelity + Send + Sync,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + Debug + Send + Sync,
    St: Stop + Send + Sync,
    Out: FidOutcome + Send + Sync,
    FnState: FuncState + Send + Sync,
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: ThrCheckpointer,
    Check::FnStateCheck: Send + Sync,
{
    /// Create a new [`ThrExperiment`] from a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// [`Stepped`], [`BatchOptimizer`], [`Stop`] condition and optional [`Recorder`] and [`Checkpointer`].
    /// It also uses an internal [`FidThrBatchEvaluator`] to evaluate a batch of [`Uncomputed`] + [`HasStep`] + [`HasFidelity`]
    /// in parallel.
    fn new(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
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
                c.init_thr();
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

    /// Load a [`ThrExperiment`] from a saved state using a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// and [`Stepped`], along with an optional [`Recorder`] and non-optional [`Checkpointer`].
    /// You can use [`load!`](crate::load) macro to load an experiment more easily.
    /// All [`FuncState`]s saved in the checkpoint will be restored in the loaded experiment.
    fn load(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        checkpointer.before_load_thr();
        let fn_check = checkpointer.new_func_state_checkpointer();

        let opt_state = checkpointer.load_optimizer_thr().unwrap();
        let stop = checkpointer.load_stop_thr().unwrap();
        let mut evaluator: FidThrBatchEvaluator<_,_,_,_,_,_> = checkpointer.load_all_evaluate_thr().unwrap().pop().expect("Only one evaluator state should be within the checkpoint for batched ThrExperiment");
        let states = fn_check.load_all_func_state();
        evaluator.pool = Arc::new(Mutex::new(IdxMapPool::from_iter(states)));

        let optimizer = Op::from_state(opt_state);
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
    /// Run the [`ThrExperiment`], performing optimization, using a [`BatchOptimizer`], until the [`Stop`] condition is met.
    /// The process evaluates [`Batch`](crate::Batch)es of [`Uncomputed`] + [`HasStep`] + [`HasFidelity`] using the inner [`FidThrBatchEvaluator`],
    /// A checkpoint is performed after each optimization step. And [`CompBatch`](crate::Batch)es of [`Computed`](crate::Computed),
    /// are saved using the inner [`Recorder`] when [`FidThrBatchEvaluator`] has finished evaluating all elements.
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`FidBatchEvaluator`] updates) step.
    fn run(mut self) {
        let ob = Arc::new(self.objective);
        let cod = Arc::new(self.codomain);
        let st = Arc::new(Mutex::new(self.stop));
        let scp = Arc::new(self.searchspace);

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let fn_check = self
                    .checkpointer
                    .as_ref()
                    .map(|c| c.new_func_state_checkpointer());
                FidThrBatchEvaluator::new(
                    self.optimizer.first_step(&scp),
                    IdxMapPool::new(fn_check),
                )
            }
        };

        let mut batch;
        let mut computed;
        let mut outputed;

        st.lock().unwrap().init();
        'main: loop {
            // Stop part
            {
                let mut stlock = st.lock().unwrap();
                if stlock.stop() {
                    break 'main;
                };
                stlock.update(ExpStep::Iteration);
            }
            // Evaluation part
            (computed, outputed) =
                ThrEvaluate::<
                    _,
                    SId,
                    Op,
                    Scp,
                    Out,
                    St,
                    _,
                    OutBatchEvaluate<SId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
                >::evaluate(&mut eval, ob.clone(), cod.clone(), st.clone());

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch(&computed, &outputed, &scp, &cod);
            }

            // Optimizer part
            batch = self.optimizer.step(computed, &scp);
            eval.update(batch);

            // Stop and checkpointing part
            {
                let mut stlock = st.lock().unwrap();
                stlock.update(ExpStep::Optimization);
                if let Some(c) = &self.checkpointer {
                    c.save_state_thr(self.optimizer.get_mut_state(), &*stlock, &eval, 0)
                }
            }
        }
    }

    fn get_stop(&self) -> &St {
        &self.stop
    }

    fn get_searchspace(&self) -> &Scp {
        &self.searchspace
    }

    fn get_codomain(&self) -> &Op::Cod {
        &self.codomain
    }

    fn get_objective(&self) -> &Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
        &self.objective
    }

    fn get_optimizer(&self) -> &Op {
        &self.optimizer
    }

    fn get_recorder(&self) -> Option<&Rec> {
        self.recorder.as_ref()
    }

    fn get_checkpointer(&self) -> Option<&Check> {
        self.checkpointer.as_ref()
    }

    fn get_mut_stop(&mut self) -> &mut St {
        &mut self.stop
    }

    fn get_mut_searchspace(&mut self) -> &mut Scp {
        &mut self.searchspace
    }

    fn get_mut_codomain(&mut self) -> &mut Op::Cod {
        &mut self.codomain
    }

    fn get_mut_objective(
        &mut self,
    ) -> &mut Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
        &mut self.objective
    }

    fn get_mut_optimizer(&mut self) -> &mut Op {
        &mut self.optimizer
    }

    fn get_mut_recorder(&mut self) -> Option<&mut Rec> {
        self.recorder.as_mut()
    }

    fn get_mut_checkpointer(&mut self) -> Option<&mut Check> {
        self.checkpointer.as_mut()
    }
}

//-------------------//
//--- DISTRIBUTED ---//
//-------------------//

#[cfg(feature = "mpi")]
impl<'a, PSol, Scp, Op, St, Rec, Check, Out>
    MPIRunable<
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
    for MPIExperiment<
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
    /// Describes the [`Worker`](crate::Worker) type used in the distributed experiment.
    /// Here a simple [`BaseWorker`] is used with an inner [`Objective`] and [`MPIProcess`].
    type WType = BaseWorker<'a, RawObj<Scp::SolShape, SId, Op::SInfo>, Out>;

    /// Create a new distributed [`MPIExperiment`] wrapped in a [`MasterWorker`] from a [`Searchspace`],
    /// [`Codomain`](crate::Codomain), [`Objective`], [`BatchOptimizer`], [`Stop`] condition and optional
    /// [`DistRecorder`] and [`DistCheckpointer`]. The main process (rank 0) will be the [`Master`](crate::MasterWorker) while
    /// all other processes will be [`Worker`](crate::Worker)s.
    /// It also uses an internal [`BatchEvaluator`] to evaluate a batch of [`Uncomputed`] in parallel across the distributed processes.
    /// The [`DistRecorder`] and [`DistCheckpointer`] are only used by the main process.
    /// Other processes will use a [`NoWCheck`](crate::checkpointer::NoWCheck) version of the [`DistCheckpointer`].
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
            MasterWorker::Master(MPIExperiment {
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

    /// Load a distributed [`MPIExperiment`] wrapped in a [`MasterWorker`] from a [`Searchspace`],
    /// [`Codomain`](crate::Codomain), [`Objective`], along with an optional [`DistRecorder`] and non-optional [`DistCheckpointer`].
    /// The main process (rank 0) will be the [`Master`](crate::MasterWorker) loaded via [`load_dist`](crate::DistCheckpointer::load_dist)
    /// while all other processes will be [`Worker`](crate::Worker)s loaded here via [`no_check_init`](crate::DistCheckpointer::no_check_init).
    /// The loading process follows the logic described in the [`DistCheckpointer`]
    /// concrete implementations (e.g. [`MessagePack`](crate::checkpointer::MessagePack)).
    ///
    /// You can use [`load!`](crate::load) macro to load an experiment more easily.
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
            checkpointer.before_load_dist(proc);

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

            MasterWorker::Master(MPIExperiment {
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

    /// Run the distributed [`MPIExperiment`], performing optimization, using a [`BatchOptimizer`], until the [`Stop`] condition is met.
    /// The main process (rank 0) will coordinate the optimization using the inner [`BatchEvaluator`]
    /// to evaluate [`Batch`](crate::Batch)es of [`Uncomputed`] across all processes.
    /// [`Batch`](crate::Batch)es are obtained via the [`step`](crate::BatchOptimizer::step) only processed by rank 0.
    /// A checkpoint is performed after each optimization step by the main process. The main process sends a
    /// checkpoint order to all idle processes via the internal [`SendRec`] utility.
    /// And also a stop order is sent to all processes when the experiment is finished, allowing for a clean
    /// termination of all [`Worker`](crate::Worker)s.
    /// And [`CompBatch`](crate::Batch)es of [`Computed`](crate::Computed), are saved using the inner [`DistRecorder`], performed
    /// by rank 0 when [`BatchEvaluator`] has finished evaluating all elements.
    ///
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`BatchEvaluator`] updates) step by the main process.
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

        self.stop.init();
        'main: loop {
            // Stop part
            if self.stop.stop() {
                break 'main;
            };
            self.stop.update(ExpStep::Iteration); // New iteration

            // Evaluation part
            (computed, outputed) = DistEvaluate::<
                _,
                SId,
                Op,
                Scp,
                Out,
                St,
                _,
                _,
                OutBatchEvaluate<SId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
            >::evaluate(
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

            // Optimizer part
            batch = self.optimizer.step(computed, &self.searchspace);
            eval.update(batch);
            self.stop.update(ExpStep::Optimization);

            // Checkpointing part
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(
                    self.optimizer.get_mut_state(),
                    &self.stop,
                    &eval,
                    self.proc.rank,
                );
            }
        }
        stop_order(self.proc, 1..self.proc.size);
    }

    fn get_stop(&self) -> &St {
        &self.stop
    }

    fn get_searchspace(&self) -> &Scp {
        &self.searchspace
    }

    fn get_codomain(&self) -> &Op::Cod {
        &self.codomain
    }

    fn get_objective(&self) -> &Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out> {
        &self.objective
    }

    fn get_optimizer(&self) -> &Op {
        &self.optimizer
    }

    fn get_recorder(&self) -> Option<&Rec> {
        self.recorder.as_ref()
    }

    fn get_checkpointer(&self) -> Option<&Check> {
        self.checkpointer.as_ref()
    }

    fn get_mut_stop(&mut self) -> &mut St {
        &mut self.stop
    }

    fn get_mut_searchspace(&mut self) -> &mut Scp {
        &mut self.searchspace
    }

    fn get_mut_codomain(&mut self) -> &mut Op::Cod {
        &mut self.codomain
    }

    fn get_mut_objective(&mut self) -> &mut Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out> {
        &mut self.objective
    }

    fn get_mut_optimizer(&mut self) -> &mut Op {
        &mut self.optimizer
    }

    fn get_mut_recorder(&mut self) -> Option<&mut Rec> {
        self.recorder.as_mut()
    }

    fn get_mut_checkpointer(&mut self) -> Option<&mut Check> {
        self.checkpointer.as_mut()
    }
}

#[cfg(feature = "mpi")]
impl<'a, PSol, Scp, Op, St, Rec, Check, Out, FnState>
    MPIRunable<
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
    for MPIExperiment<
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
    /// Describes the [`Worker`](crate::Worker) type used in the distributed experiment.
    /// Here a [`FidWorker`] is used with an inner [`Stepped`] and [`MPIProcess`].
    /// It handles internal [`FuncState`] management for fidelity-based evaluations.
    type WType = FidWorker<'a, SId, RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState, Check>;

    /// Create a new distributed [`MPIExperiment`] wrapped in a [`MasterWorker`] from a [`Searchspace`],
    /// [`Codomain`](crate::Codomain), [`Stepped`], [`BatchOptimizer`], [`Stop`] condition and optional
    /// [`DistRecorder`] and [`DistCheckpointer`]. The main process (rank 0) will be the [`Master`](crate::MasterWorker) while
    /// all other processes will be [`Worker`](crate::Worker)s.
    /// It also uses an internal [`FidDistBatchEvaluator`] to evaluate a batch of [`Uncomputed`] + [`HasStep`] + [`HasFidelity`]
    /// in parallel across the distributed processes.
    /// The [`DistRecorder`] and [`DistCheckpointer`] are only used by the main process.
    /// Other processes will use a [`WorkerCheckpointer`](crate::checkpointer::WorkerCheckpointer) version of the [`DistCheckpointer`].
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
            MasterWorker::Master(MPIExperiment {
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
                    wc.init(proc);
                    Some(wc)
                }
                None => None,
            };
            MasterWorker::Worker(FidWorker::new(objective, check, proc))
        }
    }

    /// Load a distributed [`MPIExperiment`] wrapped in a [`MasterWorker`] from a [`Searchspace`],
    /// [`Codomain`](crate::Codomain), [`Stepped`], along with an optional [`DistRecorder`] and non-optional [`DistCheckpointer`].
    /// The main process (rank 0) will be the [`Master`](crate::MasterWorker) loaded via [`load_dist`](crate::DistCheckpointer::load_dist)
    /// while all other processes will be [`Worker`](crate::Worker)s loaded here via their respective [`WorkerCheckpointer`](crate::checkpointer::WorkerCheckpointer).
    /// The loading process follows the logic described in the [`DistCheckpointer`]
    /// concrete implementations (e.g. [`MessagePack`](crate::checkpointer::MessagePack)).
    ///
    /// # Note
    ///
    /// All [`FuncState`]s saved in the checkpoint will be restored in the loaded experiment, within the
    /// corresponding [`FidWorker`]s.
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
            checkpointer.before_load_dist(proc);
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

            MasterWorker::Master(MPIExperiment {
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
            check.before_load(proc);
            let state = check.load(proc.rank).unwrap();
            let mut w = FidWorker::new(objective, Some(check), proc);
            w.state = state;
            MasterWorker::Worker(w)
        }
    }

    /// Run the distributed [`MPIExperiment`], performing optimization, using a [`BatchOptimizer`], until the [`Stop`] condition is met.
    /// The main process (rank 0) will coordinate the optimization using the inner [`FidDistBatchEvaluator`]
    /// to evaluate [`Batch`](crate::Batch)es of [`Uncomputed`] + [`HasStep`] + [`HasFidelity`]
    /// across all processes.
    /// [`Batch`](crate::Batch)es are obtained via the [`step`](crate::BatchOptimizer::step) only processed by rank 0.
    /// A checkpoint is performed after each optimization step by the main process. The main process sends a
    /// checkpoint order to all idle processes via the internal [`SendRec`] utility.
    /// And also a stop order is sent to all processes when the experiment is finished, allowing for a clean
    /// termination of all [`Worker`](crate::Worker)s.
    /// [`CompBatch`](crate::Batch)es of [`Computed`](crate::Computed), are saved using the inner [`DistRecorder`] when
    /// [`FidDistBatchEvaluator`] has finished evaluating all elements.
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

        self.stop.init();
        'main: loop {
            // Stop part
            if self.stop.stop() {
                break 'main;
            };
            self.stop.update(ExpStep::Iteration); // New iteration

            // Evaluation part
            (computed, outputed) = DistEvaluate::<
                _,
                SId,
                Op,
                Scp,
                Out,
                St,
                _,
                _,
                OutBatchEvaluate<SId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
            >::evaluate(
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

            // Optimizer part
            batch = self.optimizer.step(computed, &self.searchspace);
            eval.update(batch);
            self.stop.update(ExpStep::Optimization);

            // Checkpointing part
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(
                    self.optimizer.get_mut_state(),
                    &self.stop,
                    &eval,
                    self.proc.rank,
                );
                sendrec.checkpoint_order(); // To all idle process.
            }
        }
        stop_order(self.proc, 1..self.proc.size);
    }

    fn get_stop(&self) -> &St {
        &self.stop
    }

    fn get_searchspace(&self) -> &Scp {
        &self.searchspace
    }

    fn get_codomain(&self) -> &Op::Cod {
        &self.codomain
    }

    fn get_objective(&self) -> &Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
        &self.objective
    }

    fn get_optimizer(&self) -> &Op {
        &self.optimizer
    }

    fn get_recorder(&self) -> Option<&Rec> {
        self.recorder.as_ref()
    }

    fn get_checkpointer(&self) -> Option<&Check> {
        self.checkpointer.as_ref()
    }

    fn get_mut_stop(&mut self) -> &mut St {
        &mut self.stop
    }

    fn get_mut_searchspace(&mut self) -> &mut Scp {
        &mut self.searchspace
    }

    fn get_mut_codomain(&mut self) -> &mut Op::Cod {
        &mut self.codomain
    }

    fn get_mut_objective(
        &mut self,
    ) -> &mut Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
        &mut self.objective
    }

    fn get_mut_optimizer(&mut self) -> &mut Op {
        &mut self.optimizer
    }

    fn get_mut_recorder(&mut self) -> Option<&mut Rec> {
        self.recorder.as_mut()
    }

    fn get_mut_checkpointer(&mut self) -> Option<&mut Check> {
        self.checkpointer.as_mut()
    }
}
