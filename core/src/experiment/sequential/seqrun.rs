use crate::{
    Accumulator, Codomain, FidOutcome, SId, SeqRecorder, Solution, Stepped,
    checkpointer::{FuncStateCheckpointer, MonoCheckpointer, ThrCheckpointer},
    domain::{codomain::TypeAcc, onto::LinkOpt},
    experiment::{
        CompAcc, MonoEvaluate, MonoExperiment, OutShapeEvaluate, PoolMode, Runable, ThrExperiment,
        basics::{FuncStatePool, IdxMapPool, LoadPool, Pool},
        sequential::{
            seqevaluator::{SeqEvaluator, ThrSeqEvaluator, VecThrSeqEvaluator},
            seqfidevaluator::{FidSeqEvaluator, FidThrSeqEvaluator, PoolFidThrSeqEvaluator},
        },
    },
    objective::{Objective, Outcome, Step, outcome::FuncState},
    optimizer::opt::{OpSInfType, SequentialOptimizer},
    searchspace::{CompShape, Searchspace},
    solution::{
        HasFidelity, HasId, HasStep, IntoComputed, SolutionShape, Uncomputed, id::{StepId, StepSId}, shape::RawObj
    },
    stop::{ExpStep, Stop},
};

use std::{
    collections::VecDeque,
    ops::Deref,
    sync::{Arc, Mutex},
    thread,
};

#[cfg(feature = "mpi")]
use crate::{
    DistSeqRecorder,
    checkpointer::{DistCheckpointer, WorkerCheckpointer},
    experiment::{
        DistEvaluate, MPIExperiment, MPIRunable, MasterWorker,
        mpi::{
            utils::{FXMessage, MPIProcess, SendRec, XMessage, stop_order},
            worker::{BaseWorker, FidWorker},
        },
        sequential::{seqevaluator::DistSeqEvaluator, seqfidevaluator::FidDistSeqEvaluator},
    },
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
        SeqEvaluator<SId, Op::SInfo, Scp::SolShape>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    Op: SequentialOptimizer<
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
    Rec:
        SeqRecorder<PSol, SId, Out, Scp, Op, Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>>,
    Check: MonoCheckpointer,
    Out: Outcome,
{
    /// Create a new [`MonoExperiment`] from a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// [`Objective`], [`SequentialOptimizer`], [`Stop`] condition and optional [`Recorder`] and [`Checkpointer`].
    fn new_with_pool(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
        _pool_mode: PoolMode,
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

        let accumulator = Op::Cod::new_accumulator();

        MonoExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            accumulator,
            evaluator: None,
            pool_mode: _pool_mode,
        }
    }

    /// Load a [`MonoExperiment`] from a saved state using a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// and [`Objective`], along with an optional [`Recorder`] and non-optional [`Checkpointer`].
    /// You can use [`load!`](crate::load) macro to load an experiment more easily.
    fn load_with_pool(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        saver: (Option<Rec>, Check),
        _pool_mode: PoolMode,
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
        let accumulator = checkpointer.load_accumulator().unwrap();
        checkpointer.after_load();

        MonoExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            accumulator,
            evaluator: Some(evaluator),
            pool_mode: _pool_mode,
        }
    }

    /// Run the [`MonoExperiment`], performing optimization, using a [`SequentialOptimizer`], until the [`Stop`] condition is met.
    /// The process evaluates a single [`SolutionShape`] of [`Uncomputed`], per iteration, using the inner [`SeqEvaluator`],
    /// A checkpoint is performed after each optimization step. And a single [`Computed`](crate::Computed),
    /// is saved using the inner [`Recorder`] when [`SeqEvaluator`] has finished evaluating an element.
    ///
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`SeqEvaluator`]. updates) step.
    fn run(mut self) -> CompAcc<Scp, PSol, SId, Op::SInfo, Op::Cod, Out> {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => SeqEvaluator::new(self.optimizer.step(
                None,
                &self.searchspace,
                &self.accumulator,
            )),
        };

        let mut computed;
        let mut outputed;
        loop {
            // Stop part
            if self.stop.stop() {
                break;
            };
            self.stop.update(ExpStep::Iteration); // New iteration

            // Evaluate the solution
            (computed, outputed) = MonoEvaluate::<
                _,
                SId,
                Op,
                Scp,
                Out,
                St,
                _,
                OutShapeEvaluate<SId, Op::SInfo, Scp, PSol, Op::Cod, Out>,
            >::evaluate(
                &mut eval,
                &self.objective,
                &self.codomain,
                &mut self.stop,
                &mut self.accumulator,
            );

            // Check if stop
            if let Some(r) = &self.recorder {
                r.save(&computed, &outputed, &self.searchspace, &self.codomain);
            }

            eval = self
                .optimizer
                .step(Some(computed), &self.searchspace, &self.accumulator)
                .into();
            self.stop.update(ExpStep::Optimization);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval);
                c.save_accumulator(&self.accumulator);
            }
        }

        self.accumulator
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

    fn get_accumalator(
        &self,
    ) -> &TypeAcc<Op::Cod, CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>, SId, Op::SInfo, Out>
    {
        &self.accumulator
    }

    fn get_mut_accumalator(
        &mut self,
    ) -> &mut TypeAcc<
        Op::Cod,
        CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>,
        SId,
        Op::SInfo,
        Out,
    > {
        &mut self.accumulator
    }
    
    fn extract(self) -> ((Scp, Op::Cod),Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,Op,St,(Option<Rec>, Option<Check>)) {
        ((self.searchspace, self.codomain), self.objective, self.optimizer, self.stop, (self.recorder, self.checkpointer))
    }
}

impl<PSol, Scp, Op, St, Rec, Check, Out, FnState>
    Runable<
        PSol,
        StepSId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
    >
    for MonoExperiment<
        PSol,
        StepSId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
        FidSeqEvaluator<
            StepSId,
            Op::SInfo,
            Scp::SolShape,
            FnState,
            Pool<Check::FnStateCheck, FnState, StepSId>,
        >,
    >
where
    PSol: Uncomputed<StepSId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity,
    Op: SequentialOptimizer<
            PSol,
            StepSId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<RawObj<Scp::SolShape, StepSId, OpSInfType<Op, PSol, Scp, StepSId, Out>>, Out, FnState>,
        >,
    Scp: Searchspace<PSol, StepSId, OpSInfType<Op, PSol, Scp, StepSId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    CompShape<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<StepSId, Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Rec: SeqRecorder<
            PSol,
            StepSId,
            Out,
            Scp,
            Op,
            Stepped<RawObj<Scp::SolShape, StepSId, OpSInfType<Op, PSol, Scp, StepSId, Out>>, Out, FnState>,
        >,
    Check: MonoCheckpointer,
    Out: FidOutcome,
    FnState: FuncState,
{
    /// Create a new [`MonoExperiment`] from a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// [`Objective`], [`SequentialOptimizer`], [`Stop`] condition and optional [`Recorder`] and [`Checkpointer`].
    fn new_with_pool(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
        pool_mode: PoolMode,
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
        let accumulator = Op::Cod::new_accumulator();
        MonoExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            accumulator,
            evaluator: None,
            pool_mode,
        }
    }

    /// Load a [`MonoExperiment`] from a saved state using a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// and [`Stepped`], along with an optional [`Recorder`] and non-optional [`Checkpointer`].
    /// You can use [`load!`](crate::load) macro to load an experiment more easily.
    fn load_with_pool(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
        saver: (Option<Rec>, Check),
        pool_mode: PoolMode,
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        checkpointer.before_load();
        let fnstatecheck = checkpointer.new_func_state_checkpointer();

        let opt_state = checkpointer.load_optimizer().unwrap();
        let stop = checkpointer.load_stop().unwrap();
        let mut evaluator: FidSeqEvaluator<_, _, _, _, _> = checkpointer.load_evaluate().unwrap();

        match pool_mode {
            PoolMode::InMemory => {
                let fn_states = fnstatecheck.load_all_func_state();
                let mut pool = IdxMapPool::from_iter(fn_states);
                pool.check = Some(fnstatecheck);
                evaluator.pool = Pool::IdxMap(pool);
            }
            PoolMode::Persistent => {
                evaluator.pool = Pool::Load(LoadPool::new(fnstatecheck));
            }
        }

        let optimizer = Op::from_state(opt_state);
        let recorder = match recorder {
            Some(mut r) => {
                r.after_load(&searchspace, &codomain);
                Some(r)
            }
            None => None,
        };

        let accumulator = checkpointer.load_accumulator().unwrap();

        checkpointer.after_load();

        MonoExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            accumulator,
            evaluator: Some(evaluator),
            pool_mode,
        }
    }

    /// Run the [`MonoExperiment`], performing optimization, using a [`SequentialOptimizer`], until the [`Stop`] condition is met.
    /// The process evaluates a single [`SolutionShape`] of [`Uncomputed`] + [`HasStep`] + [`HasFidelity`],
    /// per iteration, using the inner [`FidSeqEvaluator`].
    /// A checkpoint is performed after each optimization step. And a single [`Computed`](crate::Computed),
    /// is saved using the inner [`Recorder`] when [`FidSeqEvaluator`] has finished evaluating an element.
    ///
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`FidSeqEvaluator`]. updates) step.
    fn run(mut self) -> CompAcc<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out> {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let fn_check = self
                    .checkpointer
                    .as_ref()
                    .map(|c| c.new_func_state_checkpointer());
                let sol = self
                    .optimizer
                    .step(None, &self.searchspace, &self.accumulator)
                    .into();
                match self.pool_mode {
                    PoolMode::InMemory => {
                        let pool = IdxMapPool::new(fn_check);
                        FidSeqEvaluator::new(sol, Pool::IdxMap(pool))
                    }
                    PoolMode::Persistent => {
                        FidSeqEvaluator::new(sol, Pool::Load(LoadPool::new(fn_check.unwrap())))
                    }
                }
            }
        };

        self.stop.init();
        loop {
            // Stop part
            if self.stop.stop() {
                break;
            };
            self.stop.update(ExpStep::Iteration); // New iteration

            // Evaluate the solution
            let output = MonoEvaluate::<
                _,
                StepSId,
                Op,
                Scp,
                Out,
                St,
                _,
                Option<OutShapeEvaluate<StepSId, Op::SInfo, Scp, PSol, Op::Cod, Out>>,
            >::evaluate(
                &mut eval,
                &self.objective,
                &self.codomain,
                &mut self.stop,
                &mut self.accumulator,
            );

            let (computed, outputed) = output.unzip();

            // Saver part
            if let Some(comp) = &computed
                && let Some(out) = &outputed
                && let Some(r) = &self.recorder
            {
                r.save(comp, out, &self.searchspace, &self.codomain);
            }

            eval.update(
                self.optimizer
                    .step(computed, &self.searchspace, &self.accumulator),
            );
            self.stop.update(ExpStep::Optimization);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval);
                c.save_accumulator(&self.accumulator);
            }
        }
        self.accumulator
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

    fn get_objective(&self) -> &Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState> {
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
    ) -> &mut Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState> {
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

    fn get_accumalator(
        &self,
    ) -> &TypeAcc<Op::Cod, CompShape<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out>, StepSId, Op::SInfo, Out>
    {
        &self.accumulator
    }

    fn get_mut_accumalator(
        &mut self,
    ) -> &mut TypeAcc<
        Op::Cod,
        CompShape<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out>,
        StepSId,
        Op::SInfo,
        Out,
    > {
        &mut self.accumulator
    }
    
    fn extract(self) -> ((Scp, Op::Cod),Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,Op,St,(Option<Rec>, Option<Check>)) {
        ((self.searchspace, self.codomain), self.objective, self.optimizer, self.stop, (self.recorder, self.checkpointer))
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
        VecThrSeqEvaluator<Scp::SolShape, SId, Op::SInfo>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo> + 'static,
    Op: SequentialOptimizer<
            PSol,
            SId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Objective<RawObj<Scp::SolShape, SId, OpSInfType<Op, PSol, Scp, SId, Out>>, Out>,
        > + Send
        + Sync
        + 'static,
    Op::Cod: Send + Sync + 'static,
    Op::SInfo: Send + Sync + 'static,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>> + Send + Sync + 'static,
    Scp::SolShape: Send + Sync + 'static,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + Send + Sync + 'static,
    St: Stop + Send + Sync + 'static,
    Out: Outcome + Send + Sync + 'static,
    Rec: SeqRecorder<PSol, SId, Out, Scp, Op, Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>>
        + Send
        + Sync
        + 'static,
    Check: ThrCheckpointer + Send + Sync + 'static,
    TypeAcc<Op::Cod, CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>, SId, Op::SInfo, Out>:
        Send + Sync + 'static,
{
    /// Create a new [`ThrExperiment`] from a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// [`Objective`], [`SequentialOptimizer`], [`Stop`] condition and optional [`Recorder`] and [`Checkpointer`].
    fn new_with_pool(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
        _pool_mode: PoolMode,
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
        let accumulator = Op::Cod::new_accumulator();

        ThrExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            accumulator,
            evaluator: None,
            pool_mode: PoolMode::InMemory,
        }
    }

    /// Load a [`ThrExperiment`] from a saved state using a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// and [`Objective`], along with an optional [`Recorder`] and non-optional [`Checkpointer`].
    /// You can use [`load!`](crate::load) macro to load an experiment more easily.
    fn load_with_pool(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        saver: (Option<Rec>, Check),
        _pool_mode: PoolMode,
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        checkpointer.before_load_thr();
        let stop = checkpointer.load_stop_thr().unwrap();
        let opt_state = checkpointer.load_optimizer_thr().unwrap();
        let evaluators: Vec<ThrSeqEvaluator<_, _, _>> =
            checkpointer.load_all_evaluate_thr().unwrap();

        let evaluator = evaluators.into();
        let optimizer = Op::from_state(opt_state);
        let recorder = match recorder {
            Some(mut rec) => {
                rec.after_load(&searchspace, &codomain);
                Some(rec)
            }
            None => None,
        };
        let accumulator = checkpointer.load_accumulator_thr().unwrap();

        checkpointer.after_load();

        ThrExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            accumulator,
            evaluator: Some(evaluator),
            pool_mode: _pool_mode,
        }
    }

    /// Run the [`ThrExperiment`], performing optimization, using a [`SequentialOptimizer`], until the [`Stop`] condition is met.
    /// Each thread evaluates a single [`SolutionShape`] of [`Uncomputed`] using the inner [`ThrSeqEvaluator`],
    /// while asking on demand new solutions from the shared [`SequentialOptimizer`].
    /// A checkpoint is performed after each optimization step. And a single [`Computed`](crate::Computed),
    /// is saved using the inner [`Recorder`] when [`ThrSeqEvaluator`] has finished evaluating an element.
    ///
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`ThrSeqEvaluator`]. updates) step.
    fn run(self) -> CompAcc<Scp, PSol, SId, Op::SInfo, Op::Cod, Out> {
        let ob = Arc::new(self.objective);
        let op = Arc::new(Mutex::new(self.optimizer));
        let cod = Arc::new(self.codomain);
        let st = Arc::new(Mutex::new(self.stop));
        let scp = Arc::new(self.searchspace);
        let checkpointer = Arc::new(self.checkpointer);
        let recorder = Arc::new(self.recorder);
        let accumulator = Arc::new(Mutex::new(self.accumulator));
        let evaluator = match self.evaluator {
            Some(e) => Arc::new(Mutex::new(e)),
            None => Arc::new(Mutex::new(VecThrSeqEvaluator::new(VecDeque::new()))),
        };

        let k = num_cpus::get();
        let mut workers = Vec::with_capacity(k);

        st.lock().unwrap().init();
        for thr in 0..k {
            let optimizer = op.clone();
            let objective = ob.clone();
            let scp = scp.clone();
            let cod = cod.clone();
            let stop = st.clone();
            let check = checkpointer.clone();
            let rec = recorder.clone();
            let acc = accumulator.clone();
            let evaluator = evaluator.clone();

            let handle = thread::spawn(move || {
                let mut eval = match evaluator.lock().unwrap().get_one_evaluator() {
                    Some(e) => e,
                    None => ThrSeqEvaluator::new(optimizer.lock().unwrap().step(
                        None,
                        &scp,
                        acc.lock().unwrap().deref(),
                    )),
                };
                loop {
                    if stop.lock().unwrap().stop() {
                        break;
                    }
                    stop.lock().unwrap().update(ExpStep::Iteration);

                    let pair = eval.pair.take().expect(
                        "The pair ThrSeqEvaluator should not be empty (None) during evaluate.",
                    );
                    let id = pair.id();
                    // No saved state
                    let out = objective.compute(pair.get_sobj().clone_x());
                    let y = cod.get_elem(&out);
                    let computed = pair.into_computed(y.into());
                    let outputed = (id, out);
                    stop.lock()
                        .unwrap()
                        .update(ExpStep::Distribution(Step::Evaluated));

                    if let Some(r) = rec.as_ref() {
                        r.save(&computed, &outputed, &scp, &cod);
                    }

                    eval = match evaluator.lock().unwrap().get_one_evaluator() {
                        Some(e) => e,
                        None => ThrSeqEvaluator::new(optimizer.lock().unwrap().step(
                            Some(computed),
                            &scp,
                            acc.lock().unwrap().deref(),
                        )),
                    };
                    stop.lock().unwrap().update(ExpStep::Optimization);

                    if let Some(c) = check.as_ref() {
                        c.save_state_thr(
                            optimizer.lock().unwrap().get_state(),
                            &*stop.lock().unwrap(),
                            &eval,
                            thr,
                        );
                        c.save_accumulator_thr(&*acc.lock().unwrap());
                    }
                }
            });
            workers.push(handle);
        }
        for worker in workers {
            let _ = worker.join();
        }
        Arc::try_unwrap(accumulator).unwrap().into_inner().unwrap()
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

    fn get_accumalator(
        &self,
    ) -> &TypeAcc<Op::Cod, CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>, SId, Op::SInfo, Out>
    {
        &self.accumulator
    }

    fn get_mut_accumalator(
        &mut self,
    ) -> &mut TypeAcc<
        Op::Cod,
        CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>,
        SId,
        Op::SInfo,
        Out,
    > {
        &mut self.accumulator
    }
    
    fn extract(self) -> ((Scp, Op::Cod),Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,Op,St,(Option<Rec>, Option<Check>)) {
        ((self.searchspace, self.codomain), self.objective, self.optimizer, self.stop, (self.recorder, self.checkpointer))
    }
}

impl<PSol, Scp, Op, St, Rec, Check, Out, FnState>
    Runable<
        PSol,
        StepSId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
    >
    for ThrExperiment<
        PSol,
        StepSId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
        PoolFidThrSeqEvaluator<
            Scp::SolShape,
            StepSId,
            Op::SInfo,
            FnState,
            Pool<Check::FnStateCheck, FnState, StepSId>,
        >,
    >
where
    PSol: Uncomputed<StepSId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity,
    Op: SequentialOptimizer<
            PSol,
            StepSId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<RawObj<Scp::SolShape, StepSId, OpSInfType<Op, PSol, Scp, StepSId, Out>>, Out, FnState>,
        > + Send
        + Sync
        + 'static,
    Op::Cod: Send + Sync + 'static,
    Op::SInfo: Send + Sync + 'static,
    Scp: Searchspace<PSol, StepSId, OpSInfType<Op, PSol, Scp, StepSId, Out>> + Send + Sync + 'static,
    Scp::SolShape: HasStep + HasFidelity + Send + Sync + 'static,
    CompShape<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<StepSId, Op::SInfo> + HasStep + HasFidelity + Send + Sync + 'static,
    St: Stop + Send + Sync + 'static,
    Out: FidOutcome + Send + Sync + 'static,
    Rec: SeqRecorder<
            PSol,
            StepSId,
            Out,
            Scp,
            Op,
            Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
        > + Send
        + Sync
        + 'static,
    Check: ThrCheckpointer + Send + Sync + 'static,
    Check::FnStateCheck: Send + Sync + 'static,
    FnState: FuncState + Send + Sync + 'static,
    RawObj<Scp::SolShape, StepSId, Op::SInfo>: 'static,
    TypeAcc<Op::Cod, CompShape<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out>, StepSId, Op::SInfo, Out>:
        Send + Sync + 'static,
{
    /// Create a new [`ThrExperiment`] from a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// [`Stepped`], [`SequentialOptimizer`], [`Stop`] condition and optional [`Recorder`] and [`Checkpointer`].
    fn new_with_pool(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
        pool_mode: PoolMode,
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
        let accumulator = Op::Cod::new_accumulator();
        ThrExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            accumulator,
            evaluator: None,
            pool_mode,
        }
    }

    /// Load a [`ThrExperiment`] from a saved state using a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// and [`Objective`], along with an optional [`Recorder`] and non-optional [`Checkpointer`].
    /// You can use [`load!`](crate::load) macro to load an experiment more easily.
    fn load_with_pool(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
        saver: (Option<Rec>, Check),
        pool_mode: PoolMode,
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        checkpointer.before_load_thr();
        let fnstatecheck = checkpointer.new_func_state_checkpointer();

        let stop = checkpointer.load_stop_thr().unwrap();
        let opt_state = checkpointer.load_optimizer_thr().unwrap();
        let evaluators: Vec<FidThrSeqEvaluator<Scp::SolShape, StepSId, Op::SInfo, FnState>> =
            checkpointer.load_all_evaluate_thr().unwrap();

        let pool = match pool_mode {
            PoolMode::InMemory => {
                let fn_states = fnstatecheck.load_all_func_state();
                let mut pool = IdxMapPool::from_iter(fn_states);
                pool.check = Some(fnstatecheck);
                Pool::IdxMap(pool)
            }
            PoolMode::Persistent => Pool::Load(LoadPool::new(fnstatecheck)),
        };

        let evaluator = (evaluators, pool).into();
        let optimizer = Op::from_state(opt_state);
        let recorder = match recorder {
            Some(mut rec) => {
                rec.after_load(&searchspace, &codomain);
                Some(rec)
            }
            None => None,
        };
        let accumulator = checkpointer.load_accumulator_thr().unwrap();
        checkpointer.after_load();

        ThrExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            accumulator,
            evaluator: Some(evaluator),
            pool_mode,
        }
    }

    /// Run the [`ThrExperiment`], performing optimization, using a [`SequentialOptimizer`], until the [`Stop`] condition is met.
    /// Each thread evaluates a single [`SolutionShape`] of [`Uncomputed`] + [`HasStep`] + [`HasFidelity`], using the inner [`FidThrSeqEvaluator`],
    /// while asking on demand new solutions from the shared [`SequentialOptimizer`].
    /// A checkpoint is performed after each optimization step. And a single [`Computed`](crate::Computed),
    /// is saved using the inner [`Recorder`] when [`FidThrSeqEvaluator`] has finished evaluating an element.
    ///
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`FidThrSeqEvaluator`]. updates) step.
    fn run(self) -> CompAcc<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out> {
        let ob = Arc::new(self.objective);
        let op = Arc::new(Mutex::new(self.optimizer));
        let cod = Arc::new(self.codomain);
        let st = Arc::new(Mutex::new(self.stop));
        let scp = Arc::new(self.searchspace);
        let pool_evaluator = match self.evaluator {
            Some(e) => Arc::new(Mutex::new(e)),
            None => {
                let fn_check = self
                    .checkpointer
                    .as_ref()
                    .map(|c| c.new_func_state_checkpointer());
                match self.pool_mode {
                    PoolMode::InMemory => {
                        let pool = IdxMapPool::new(fn_check);
                        Arc::new(Mutex::new(PoolFidThrSeqEvaluator::new(
                            VecDeque::new(),
                            Pool::IdxMap(pool),
                        )))
                    }
                    PoolMode::Persistent => Arc::new(Mutex::new(PoolFidThrSeqEvaluator::new(
                        VecDeque::new(),
                        Pool::Load(LoadPool::new(fn_check.unwrap())),
                    ))),
                }
            }
        };
        let recorder = Arc::new(self.recorder);
        let checkpointer = Arc::new(self.checkpointer);
        let accumulator = Arc::new(Mutex::new(self.accumulator));

        let k = num_cpus::get();
        let mut workers = Vec::with_capacity(k);

        st.lock().unwrap().init();
        for thr in 0..k {
            let optimizer = op.clone();
            let objective = ob.clone();
            let scp = scp.clone();
            let cod = cod.clone();
            let stop = st.clone();
            let check = checkpointer.clone();
            let rec = recorder.clone();
            let acc = accumulator.clone();
            let pool_evaluator = pool_evaluator.clone();

            let handle = thread::spawn(move || {
                let mut eval = match pool_evaluator.lock().unwrap().get_one_evaluator() {
                    Some(e) => e,
                    None => FidThrSeqEvaluator::new(
                        optimizer
                            .lock()
                            .unwrap()
                            .step(None, &scp, acc.lock().unwrap().deref()),
                        None,
                    ),
                };
                loop {
                    if stop.lock().unwrap().stop() {
                        break;
                    }
                    stop.lock().unwrap().update(ExpStep::Iteration);

                    let mut pair = eval.pair.take().expect(
                        "The pair ThrSeqEvaluator should not be empty (None) during evaluate.",
                    );
                    
                    let step = pair.step();
                    match step {
                        Step::Pending | Step::Partially(_) => {
                            let state = eval.state.take();
                            let fid = pair.fidelity();
                            let (out, state) =
                                objective.compute(pair.get_sobj().clone_x(), fid, state);

                            let new_step = out.get_step().into();
                            pair.mut_ref_id().increment();
                            let id = pair.id();

                            match new_step {
                                Step::Partially(_) => {
                                    pool_evaluator.lock().unwrap().pool.remove(&id.previous_id());
                                    pool_evaluator.lock().unwrap().pool.insert(id, state);
                                }
                                _ => {
                                    pool_evaluator.lock().unwrap().pool.remove(&id.previous_id());
                                }
                            }

                            let y = cod.get_elem(&out);
                            let mut computed = pair.into_computed(y.into());
                            let outputed = (id, out);
                            computed.set_step(new_step);
                            acc.lock().unwrap().accumulate(&computed);
                            stop.lock().unwrap().update(ExpStep::Distribution(new_step));

                            if let Some(r) = rec.as_ref() {
                                r.save(&computed, &outputed, &scp, &cod);
                            }

                            let new_sol = optimizer.lock().unwrap().step(
                                Some(computed),
                                &scp,
                                acc.lock().unwrap().deref(),
                            );
                            let new_eval = FidThrSeqEvaluator::new(new_sol, None);
                            pool_evaluator.lock().unwrap().pairs.push_back(new_eval);
                        }
                        _ => {
                            pool_evaluator.lock().unwrap().pool.remove(pair.ref_id());
                            stop.lock().unwrap().update(ExpStep::Distribution(step));
                        }
                    }

                    eval = match pool_evaluator.lock().unwrap().get_one_evaluator() {
                        Some(e) => e,
                        None => FidThrSeqEvaluator::new(
                            optimizer
                                .lock()
                                .unwrap()
                                .step(None, &scp, acc.lock().unwrap().deref()),
                            None,
                        ),
                    };
                    stop.lock().unwrap().update(ExpStep::Optimization);

                    if let Some(c) = check.as_ref() {
                        c.save_state_thr(
                            optimizer.lock().unwrap().get_state(),
                            &*stop.lock().unwrap(),
                            &eval,
                            thr,
                        );
                        c.save_accumulator_thr(&*acc.lock().unwrap());
                    }
                }
            });
            workers.push(handle);
        }
        for worker in workers {
            let _ = worker.join();
        }
        Arc::try_unwrap(accumulator).unwrap().into_inner().unwrap()
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

    fn get_objective(&self) -> &Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState> {
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
    ) -> &mut Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState> {
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

    fn get_accumalator(
        &self,
    ) -> &TypeAcc<Op::Cod, CompShape<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out>, StepSId, Op::SInfo, Out>
    {
        &self.accumulator
    }

    fn get_mut_accumalator(
        &mut self,
    ) -> &mut TypeAcc<
        Op::Cod,
        CompShape<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out>,
        StepSId,
        Op::SInfo,
        Out,
    > {
        &mut self.accumulator
    }
    
    fn extract(self) -> ((Scp, Op::Cod),Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,Op,St,(Option<Rec>, Option<Check>)) {
        ((self.searchspace, self.codomain), self.objective, self.optimizer, self.stop, (self.recorder, self.checkpointer))
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
        DistSeqEvaluator<SId, Op::SInfo, Scp::SolShape>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    Op: SequentialOptimizer<
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
    Rec: DistSeqRecorder<
            PSol,
            SId,
            Out,
            Scp,
            Op,
            Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        >,
    Check: DistCheckpointer,
    Out: Outcome,
{
    /// Describes the [`Worker`](crate::Worker) type used in the distributed experiment.
    /// Here a simple [`BaseWorker`] is used with an inner [`Objective`] and [`MPIProcess`].
    type WType = BaseWorker<'a, RawObj<Scp::SolShape, SId, Op::SInfo>, Out>;

    /// Create a new distributed [`MPIExperiment`] wrapped in a [`MasterWorker`] from a [`Searchspace`],
    /// [`Codomain`](crate::Codomain), [`Objective`], [`SequentialOptimizer`], [`Stop`] condition and optional
    /// [`DistRecorder`] and [`DistCheckpointer`]. The main process (rank 0) will be the [`Master`](crate::MasterWorker) while
    /// all other processes will be [`Worker`](crate::Worker)s.
    /// It also uses an internal [`DistSeqEvaluator`] to evaluate single [`SolutionShape`]s per process.
    /// The [`DistRecorder`] and [`DistCheckpointer`] are only used by the main process.
    /// Other processes will use a [`NoWCheck`](crate::checkpointer::NoWCheck) version of the [`DistCheckpointer`].
    fn new_with_pool(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
        _pool_mode: PoolMode,
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
            let accumulator = Op::Cod::new_accumulator();
            MasterWorker::Master(MPIExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer,
                accumulator,
                evaluator: None,
                pool_mode: PoolMode::InMemory,
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
    fn load_with_pool(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        saver: (Option<Rec>, Check),
        _pool_mode: PoolMode,
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
            let accumulator = Op::Cod::new_accumulator();
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
                accumulator,
                evaluator: Some(evaluator),
                pool_mode: PoolMode::InMemory,
            })
        } else {
            <Check as DistCheckpointer>::no_check_init(proc);
            MasterWorker::Worker(BaseWorker::new(objective, proc))
        }
    }

    /// Run the distributed [`MPIExperiment`], performing optimization, using a [`SequentialOptimizer`], until the [`Stop`] condition is met.
    /// Each process evaluates a single [`SolutionShape`] of [`Uncomputed`] using the inner [`DistSeqEvaluator`],
    /// while asking on demand new solutions to the Master process which computes a [`SequentialOptimizer`].
    /// A checkpoint is performed after each optimization step by the main process (rank 0). And a single [`Computed`](crate::Computed),
    /// is saved using the inner [`DistRecorder`] when [`DistSeqEvaluator`] has finished evaluating an element.
    ///
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`DistSeqEvaluator`]. updates) step.
    ///
    /// At the end of the experiment, all processes are sent a stop signal, to cleanly end the distributed run.
    /// This can result in overflows of evaluated solutions after the stop condition is met, which are [`flushed`](SendRec::flush).
    fn run(mut self) -> CompAcc<Scp, PSol, SId, Op::SInfo, Op::Cod, Out> {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let shapes: Vec<_> = (0..self.proc.size)
                    .map(|_| {
                        self.optimizer
                            .step(None, &self.searchspace, &self.accumulator)
                    })
                    .collect();
                DistSeqEvaluator::new(shapes)
            }
        };
        if eval.shapes.len() < self.proc.size as usize {
            let n = self.proc.size as usize - eval.shapes.len();
            let mut shapes: Vec<_> = (0..n)
                .map(|_| {
                    self.optimizer
                        .step(None, &self.searchspace, &self.accumulator)
                })
                .collect();
            eval.shapes.append(&mut shapes);
        }

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

        let mut output;
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
            output = DistEvaluate::<_, SId, Op, Scp, Out, St, _, _, _>::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
                &mut self.accumulator,
            );
            (computed, outputed) = output.unzip();

            // Saver part
            if let Some(comp) = &computed
                && let Some(out) = &outputed
                && let Some(r) = &self.recorder
            {
                r.save_dist(comp, out, &self.searchspace, &self.codomain);
            }

            // Optimizer part
            eval.update(
                self.optimizer
                    .step(computed, &self.searchspace, &self.accumulator),
            );
            self.stop.update(ExpStep::Optimization);

            // Checkpoint part
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(
                    self.optimizer.get_state(),
                    &self.stop,
                    &eval,
                    self.proc.rank,
                );
                c.save_accumulator_dist(&self.accumulator, self.proc.rank);
            }
        }
        // Flush all waiting solution within eval.
        sendrec.waiting.drain().for_each(|(_, p)| eval.update(p));
        if let Some(c) = &self.checkpointer {
            c.save_state_dist(
                self.optimizer.get_state(),
                &self.stop,
                &eval,
                self.proc.rank,
            );
            c.save_accumulator_dist(&self.accumulator, self.proc.rank);
        }
        // Receive everything overflowing stop
        sendrec.flush();
        stop_order(self.proc, 1..self.proc.size);
        self.accumulator
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

    fn get_accumalator(
        &self,
    ) -> &TypeAcc<Op::Cod, CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>, SId, Op::SInfo, Out>
    {
        &self.accumulator
    }

    fn get_mut_accumalator(
        &mut self,
    ) -> &mut TypeAcc<
        Op::Cod,
        CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>,
        SId,
        Op::SInfo,
        Out,
    > {
        &mut self.accumulator
    }
    
    fn extract(self) -> ((Scp, Op::Cod),Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,Op,St,(Option<Rec>, Option<Check>)) {
        ((self.searchspace, self.codomain), self.objective, self.optimizer, self.stop, (self.recorder, self.checkpointer))
    }
}

#[cfg(feature = "mpi")]
impl<'a, PSol, Scp, Op, St, Rec, Check, Out, FnState>
    MPIRunable<
        'a,
        PSol,
        StepSId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
    >
    for MPIExperiment<
        'a,
        PSol,
        StepSId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
        FidDistSeqEvaluator<StepSId, Op::SInfo, Scp::SolShape>,
    >
where
    PSol: Uncomputed<StepSId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity,
    Op: SequentialOptimizer<
            PSol,
            StepSId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<RawObj<Scp::SolShape, StepSId, OpSInfType<Op, PSol, Scp, StepSId, Out>>, Out, FnState>,
        >,
    Scp: Searchspace<PSol, StepSId, OpSInfType<Op, PSol, Scp, StepSId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    SolObj<Scp::SolShape, StepSId, Op::SInfo>: HasStep + HasFidelity,
    SolOpt<Scp::SolShape, StepSId, Op::SInfo>: HasStep + HasFidelity,
    CompShape<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<StepSId, Op::SInfo> + HasY<Op::Cod, Out> + HasStep + HasFidelity,
    St: Stop,
    Rec: DistSeqRecorder<
            PSol,
            StepSId,
            Out,
            Scp,
            Op,
            Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
        >,
    Check: DistCheckpointer,
    Out: FidOutcome,
    FnState: FuncState,
{
    /// Describes the [`Worker`](crate::Worker) type used in the distributed experiment.
    /// Here a [`FidWorker`] is used with an inner [`Stepped`] and [`MPIProcess`].
    /// It stores a [`FuncState`] to resume previous [`Step::Partially`] evaluations.
    type WType = FidWorker<
        'a,
        StepSId,
        RawObj<Scp::SolShape, StepSId, Op::SInfo>,
        Out,
        FnState,
        Check,
        Pool<Check::FnStateCheck, FnState, StepSId>,
    >;

    /// Create a new distributed [`MPIExperiment`] wrapped in a [`MasterWorker`] from a [`Searchspace`],
    /// [`Codomain`](crate::Codomain), [`Stepped`], [`SequentialOptimizer`], [`Stop`] condition and optional
    /// [`DistRecorder`] and [`DistCheckpointer`]. The main process (rank 0) will be the [`Master`](crate::MasterWorker) while
    /// all other processes will be [`Worker`](crate::Worker)s.
    /// It also uses an internal [`FidDistSeqEvaluator`] to evaluate single [`SolutionShape`]s per process.
    /// The [`DistRecorder`] and [`DistCheckpointer`] are only used by the main process.
    /// Other processes will use a [`WorkerCheckpointer`](crate::checkpointer::WorkerCheckpointer) associated to the [`DistCheckpointer`].
    fn new_with_pool(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
        pool_mode: PoolMode,
    ) -> MasterWorker<
        'a,
        Self,
        PSol,
        StepSId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
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
            let accumulator = Op::Cod::new_accumulator();
            MasterWorker::Master(MPIExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer,
                accumulator,
                evaluator: None,
                pool_mode,
            })
        } else {
            let (check, fncheck) = match checkpointer {
                Some(c) => {
                    let mut wc = c.get_check_worker(proc);
                    wc.init(proc);
                    let fncheck = wc.new_func_state_checkpointer();
                    (Some(wc), Some(fncheck))
                }
                None => (None, None),
            };
            let worker = match (pool_mode, fncheck) {
                (PoolMode::InMemory, Some(fc)) => {
                    let pool = IdxMapPool::new(Some(fc));
                    FidWorker::new(proc, objective, Pool::IdxMap(pool), check)
                }
                (PoolMode::Persistent, Some(fc)) => {
                    let pool = LoadPool::new(fc);
                    FidWorker::new(proc, objective, Pool::Load(pool), check)
                }
                (PoolMode::InMemory, None) => {
                    let pool = IdxMapPool::new(None);
                    FidWorker::new(proc, objective, Pool::IdxMap(pool), check)
                }
                (PoolMode::Persistent, None) => {
                    panic!("Persistent pool mode requires a function state checkpointer.")
                }
            };
            MasterWorker::Worker(worker)
        }
    }

    /// Load a distributed [`MPIExperiment`] wrapped in a [`MasterWorker`] from a [`Searchspace`],
    /// [`Codomain`](crate::Codomain), [`Stepped`], along with an optional [`DistRecorder`] and non-optional [`DistCheckpointer`].
    /// The main process (rank 0) will be the [`Master`](crate::MasterWorker) loaded via [`load_dist`](crate::DistCheckpointer::load_dist)
    /// while all other processes will be [`Worker`](crate::Worker)s loaded here via their associated [`WorkerCheckpointer`](crate::checkpointer::WorkerCheckpointer).
    /// The loading process follows the logic described in the [`DistCheckpointer`]
    /// concrete implementations (e.g. [`MessagePack`](crate::checkpointer::MessagePack)).
    ///
    /// You can use [`load!`](crate::load) macro to load an experiment more easily.
    fn load_with_pool(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
        saver: (Option<Rec>, Check),
        pool_mode: PoolMode,
    ) -> MasterWorker<
        'a,
        Self,
        PSol,
        StepSId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,
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
            let accumulator = checkpointer.load_accumulator_dist(proc.rank).unwrap();
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
                accumulator,
                evaluator: Some(evaluator),
                pool_mode,
            })
        } else {
            let mut check = checkpointer.get_check_worker(proc);
            check.before_load(proc);
            // There is no state for FidWorker struct, pool is saved appart
            check.after_load(proc);
            let fnstatecheck = check.new_func_state_checkpointer();

            let pool = match pool_mode {
                PoolMode::InMemory => {
                    let fn_states = fnstatecheck.load_all_func_state();
                    let mut pool = IdxMapPool::from_iter(fn_states);
                    pool.check = Some(fnstatecheck);
                    Pool::IdxMap(pool)
                }
                PoolMode::Persistent => Pool::Load(LoadPool::new(fnstatecheck)),
            };
            let worker = FidWorker::new(proc, objective, pool, Some(check));
            MasterWorker::Worker(worker)
        }
    }

    /// Run the distributed [`MPIExperiment`], performing optimization, using a [`SequentialOptimizer`], until the [`Stop`] condition is met.
    /// Each process evaluates a single [`SolutionShape`] of [`Uncomputed`] + [`HasStep`] + [`HasFidelity`], using the inner [`FidDistSeqEvaluator`],
    /// while asking on demand new solutions to the Master process which computes a [`SequentialOptimizer`].
    /// A checkpoint is performed after each optimization step by the main process (rank 0). And a single [`Computed`](crate::Computed),
    /// is saved using the inner [`DistRecorder`] when [`FidDistSeqEvaluator`] has finished evaluating an element.
    ///
    /// [`FuncState`]s are used to resume partially evaluated solutions.
    ///
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`FidDistSeqEvaluator`]. updates) step.
    ///
    /// At the end of the experiment, all processes are sent a stop signal, to cleanly end the distributed run.
    /// This can result in overflows of evaluated solutions after the stop condition is met, which are [`flushed`](SendRec::flush).
    fn run(mut self) -> CompAcc<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out> {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let init_vec: Vec<_> = (0..self.proc.size + 1)
                    .map(|_| {
                        self.optimizer
                            .step(None, &self.searchspace, &self.accumulator)
                    })
                    .collect();
                FidDistSeqEvaluator::new(init_vec, self.proc.size as usize)
            }
        };

        // Define send/rec utilitaries and parameters
        let config = bincode::config::standard(); // Bytes encoding config
        let mut sendrec = SendRec::<
            '_,
            FXMessage<StepSId, RawObj<Scp::SolShape, StepSId, Op::SInfo>>,
            Scp::SolShape,
            StepSId,
            Op::SInfo,
            Op::Cod,
            Out,
        >::new(config, self.proc);

        let mut rank;
        let mut output;
        let mut solout;
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
            output = DistEvaluate::<_, StepSId, Op, Scp, Out, St, _, _, _>::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
                &mut self.accumulator,
            );
            (rank, solout) = output.unzip();
            (computed, outputed) = solout.unzip();

            // Saver part
            if let Some(comp) = &computed
                && let Some(out) = &outputed
                && let Some(r) = &self.recorder
            {
                r.save_dist(comp, out, &self.searchspace, &self.codomain);
            }

            // Optimizer part
            eval.update(
                self.optimizer
                    .step(computed, &self.searchspace, &self.accumulator),
            );
            self.stop.update(ExpStep::Optimization);
            // If more than 1 process is idle, it means that the optimizer is not able to give new solutions to at least 2 processes,
            // so the optimizer has to generate new solution ex-nihilo.
            let idle_count = sendrec.idle.count_idle();
            if idle_count > 1 {
                for _ in 0..idle_count {
                    eval.update(
                        self.optimizer
                            .step(None, &self.searchspace, &self.accumulator),
                    );
                }
                self.stop.update(ExpStep::Optimization);
            }

            // Checkpointing part
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(
                    self.optimizer.get_state(),
                    &self.stop,
                    &eval,
                    self.proc.rank,
                );
                c.save_accumulator_dist(&self.accumulator, self.proc.rank);
                if let Some(r) = rank {
                    sendrec.rank_checkpoint_order(r);
                }
            }
        }
        // Flush all waiting solution within eval.
        sendrec.waiting.drain().for_each(|(_, p)| eval.update(p));
        if let Some(c) = &self.checkpointer {
            c.save_state_dist(
                self.optimizer.get_state(),
                &self.stop,
                &eval,
                self.proc.rank,
            );
            c.save_accumulator_dist(&self.accumulator, self.proc.rank);
        }
        // Receive everything overflowing stop
        sendrec.flush();
        stop_order(self.proc, 1..self.proc.size);
        self.accumulator
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

    fn get_objective(&self) -> &Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState> {
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
    ) -> &mut Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState> {
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

    fn get_accumalator(
        &self,
    ) -> &TypeAcc<Op::Cod, CompShape<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out>, StepSId, Op::SInfo, Out>
    {
        &self.accumulator
    }

    fn get_mut_accumalator(
        &mut self,
    ) -> &mut TypeAcc<
        Op::Cod,
        CompShape<Scp, PSol, StepSId, Op::SInfo, Op::Cod, Out>,
        StepSId,
        Op::SInfo,
        Out,
    > {
        &mut self.accumulator
    }
    
    fn extract(self) -> ((Scp, Op::Cod),Stepped<RawObj<Scp::SolShape, StepSId, Op::SInfo>, Out, FnState>,Op,St,(Option<Rec>, Option<Check>)) {
        ((self.searchspace, self.codomain), self.objective, self.optimizer, self.stop, (self.recorder, self.checkpointer))
    }
}
