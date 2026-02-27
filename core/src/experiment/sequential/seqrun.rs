use std::{
    collections::VecDeque, sync::{Arc, Mutex}, thread
};

use crate::{
    Codomain, FidOutcome, SId, Solution, Stepped,
    checkpointer::{MonoCheckpointer, ThrCheckpointer},
    domain::onto::LinkOpt,
    experiment::{
        MonoEvaluate, MonoExperiment, OutShapeEvaluate, Runable, ThrExperiment, sequential::{
            seqevaluator::{SeqEvaluator, ThrSeqEvaluator, VecThrSeqEvaluator},
            seqfidevaluator::{FidSeqEvaluator, FidThrSeqEvaluator, HashFidThrSeqEvaluator},
        }
    },
    objective::{Objective, Outcome, Step, outcome::FuncState},
    optimizer::opt::{OpSInfType, SequentialOptimizer},
    recorder::Recorder,
    searchspace::{CompShape, Searchspace},
    solution::{
        HasFidelity, HasId, HasStep, IntoComputed, SolutionShape, Uncomputed, shape::RawObj,
    },
    stop::{ExpStep, Stop},
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::{DistCheckpointer, WorkerCheckpointer},
    experiment::{
        DistEvaluate, MPIExperiment, MPIRunable, MasterWorker,
        mpi::{
            utils::{FXMessage, MPIProcess, SendRec, XMessage, stop_order},
            worker::{BaseWorker, FidWorker},
        },
        sequential::seqevaluator::DistSeqEvaluator,
        sequential::seqfidevaluator::FidDistSeqEvaluator,
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
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: MonoCheckpointer,
    Out: Outcome,
{
    /// Create a new [`MonoExperiment`] from a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// [`Objective`], [`SequentialOptimizer`], [`Stop`] condition and optional [`Recorder`] and [`Checkpointer`].
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
    /// Run the [`MonoExperiment`], performing optimization, using a [`SequentialOptimizer`], until the [`Stop`] condition is met.
    /// The process evaluates a single [`SolutionShape`] of [`Uncomputed`], per iteration, using the inner [`SeqEvaluator`],
    /// A checkpoint is performed after each optimization step. And a single [`Computed`](crate::Computed),
    /// is saved using the inner [`Recorder`] when [`SeqEvaluator`] has finished evaluating an element.
    ///
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`SeqEvaluator`]. updates) step.
    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => SeqEvaluator::new(self.optimizer.step(None, &self.searchspace)),
        };

        let mut computed;
        let mut outputed;
        'main: loop {
            // Stop part
            if self.stop.stop() {
                break 'main;
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
                &mut eval, &self.objective, &self.codomain, &mut self.stop
            );

            // Check if stop
            if let Some(r) = &self.recorder {
                r.save_pair(
                    &computed,
                    &outputed,
                    &self.searchspace,
                    &self.codomain,
                    None,
                );
            }

            eval = self
                .optimizer
                .step(Some(computed), &self.searchspace)
                .into();
            self.stop.update(ExpStep::Optimization);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval)
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
        FidSeqEvaluator<SId, Op::SInfo, Scp::SolShape, FnState>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity,
    Op: SequentialOptimizer<
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
    /// [`Objective`], [`SequentialOptimizer`], [`Stop`] condition and optional [`Recorder`] and [`Checkpointer`].
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

    /// Load a [`MonoExperiment`] from a saved state using a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// and [`Stepped`], along with an optional [`Recorder`] and non-optional [`Checkpointer`].
    /// You can use [`load!`](crate::load) macro to load an experiment more easily.
    fn load(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        checkpointer.before_load();


        let opt_state = checkpointer.load_optimizer().unwrap();
        let stop = checkpointer.load_stop().unwrap();
        let mut evaluator: FidSeqEvaluator<_,_,_,_> = checkpointer.load_evaluate().unwrap();
        let states = checkpointer.load_func_state();
        evaluator.states = states;
        
        let optimizer = Op::from_state(opt_state);
        let recorder = match recorder {
            Some(mut r) => {
                r.after_load(&searchspace, &codomain);
                Some(r)
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
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
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
    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => FidSeqEvaluator::new(self.optimizer.step(None, &self.searchspace)),
        };

        self.stop.init();
        'main: loop {
            // Stop part
            if self.stop.stop() {
                break 'main;
            };
            self.stop.update(ExpStep::Iteration); // New iteration

            // Evaluate the solution
            let output = MonoEvaluate::<
                _,
                SId,
                Op,
                Scp,
                Out,
                St,
                _,
                Option<OutShapeEvaluate<SId, Op::SInfo, Scp, PSol, Op::Cod, Out>>,
            >::evaluate(
                &mut eval, &self.objective, &self.codomain, &mut self.stop
            );

            let (computed, outputed) = output.unzip();

            // Saver part
            if let Some(comp) = &computed
                && let Some(out) = &outputed
                && let Some(r) = &self.recorder
            {
                r.save_pair(comp, out, &self.searchspace, &self.codomain, None);
            }

            eval.update(self.optimizer.step(computed, &self.searchspace));
            self.stop.update(ExpStep::Optimization);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval);
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
    Op::Info: Send + Sync + 'static,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>> + Send + Sync + 'static,
    Scp::SolShape: Send + Sync + 'static,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + Send + Sync + 'static,
    St: Stop + Send + Sync + 'static,
    Out: Outcome + Send + Sync + 'static,
    Rec: Recorder<PSol, SId, Out, Scp, Op> + Send + Sync + 'static,
    Check: ThrCheckpointer + Send + Sync + 'static,
{
    /// Create a new [`ThrExperiment`] from a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// [`Objective`], [`SequentialOptimizer`], [`Stop`] condition and optional [`Recorder`] and [`Checkpointer`].
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
        let stop = checkpointer.load_stop_thr().unwrap();
        let opt_state = checkpointer.load_optimizer_thr().unwrap();
        let evaluators: Vec<(ThrSeqEvaluator<_,_,_>, _)> = 
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

    /// Run the [`ThrExperiment`], performing optimization, using a [`SequentialOptimizer`], until the [`Stop`] condition is met.
    /// Each thread evaluates a single [`SolutionShape`] of [`Uncomputed`] using the inner [`ThrSeqEvaluator`],
    /// while asking on demand new solutions from the shared [`SequentialOptimizer`].
    /// A checkpoint is performed after each optimization step. And a single [`Computed`](crate::Computed),
    /// is saved using the inner [`Recorder`] when [`ThrSeqEvaluator`] has finished evaluating an element.
    ///
    /// The [`Stop`] condition is updated after each [`ExpStep::Iteration`], [`ExpStep::Optimization`], and [`ExpStep::Distribution`]
    /// (inner [`ThrSeqEvaluator`]. updates) step.
    fn run(self) {
        let ob = Arc::new(self.objective);
        let op = Arc::new(Mutex::new(self.optimizer));
        let cod = Arc::new(self.codomain);
        let st = Arc::new(Mutex::new(self.stop));
        let scp = Arc::new(self.searchspace);
        let checkpointer = Arc::new(self.checkpointer);
        let recorder = Arc::new(self.recorder);
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
            let info = Arc::new(Op::Info::default());
            let evaluator = evaluator.clone();

            let handle = thread::spawn(move || {
                let mut eval = match evaluator.lock().unwrap().get_one_evaluator() {
                    Some(e) => e,
                    None => ThrSeqEvaluator::new(optimizer.lock().unwrap().step(None, &scp).into()),
                };
                loop {
                    if stop.lock().unwrap().stop() {
                        break;
                    }
                    stop.lock().unwrap().update(ExpStep::Iteration);

                    let pair = eval.pair.take().expect("The pair ThrSeqEvaluator should not be empty (None) during evaluate.");
                    let id = pair.id();
                    // No saved state
                    let out = objective.compute(pair.get_sobj().get_x());
                    let y = cod.get_elem(&out);
                    let computed = pair.into_computed(y.into());
                    let outputed = (id, out);
                    stop.lock().unwrap().update(ExpStep::Distribution(Step::Evaluated));

                    if let Some(r) = rec.as_ref() {
                        r.save_pair(&computed, &outputed, &scp, &cod, Some(info.clone()));
                    }

                    eval = match evaluator.lock().unwrap().get_one_evaluator() {
                        Some(e) => e,
                        None => ThrSeqEvaluator::new(optimizer.lock().unwrap().step(Some(computed), &scp)),
                    };
                    stop.lock().unwrap().update(ExpStep::Optimization);

                    if let Some(c) = check.as_ref() {
                        c.save_state_thr(
                            optimizer.lock().unwrap().get_state(),
                            &*stop.lock().unwrap(),
                            &eval,
                            thr,
                        );
                    }
                }
            });
            workers.push(handle);
        }
        for worker in workers {
            let _ = worker.join();
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
        HashFidThrSeqEvaluator<Scp::SolShape, SId, Op::SInfo, FnState>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity,
    Op: SequentialOptimizer<
            PSol,
            SId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<RawObj<Scp::SolShape, SId, OpSInfType<Op, PSol, Scp, SId, Out>>, Out, FnState>,
        > + Send
        + Sync + 'static
    ,
    Op::Cod: Send + Sync + 'static,
    Op::SInfo: Send + Sync + 'static,
    Op::Info: Send + Sync + 'static,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>> + Send + Sync + 'static,
    Scp::SolShape: HasStep + HasFidelity + Send + Sync + 'static,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + HasStep + HasFidelity + Send + Sync + 'static,
    St: Stop + Send + Sync + 'static,
    Out: FidOutcome + Send + Sync + 'static,
    Rec: Recorder<PSol, SId, Out, Scp, Op> + Send + Sync + 'static,
    Check: ThrCheckpointer + Send + Sync + 'static,
    FnState: FuncState + Send + Sync + 'static,
    RawObj<Scp::SolShape,SId,Op::SInfo>: 'static
{
    /// Create a new [`ThrExperiment`] from a [`Searchspace`], [`Codomain`](crate::Codomain),
    /// [`Stepped`], [`SequentialOptimizer`], [`Stop`] condition and optional [`Recorder`] and [`Checkpointer`].
    fn new(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
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
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        checkpointer.before_load_thr();
        let stop = checkpointer.load_stop_thr().unwrap();
        let opt_state = checkpointer.load_optimizer_thr().unwrap();
        let evaluators: Vec<(FidThrSeqEvaluator<_,_,_,_>, _)> = 
        checkpointer.load_all_evaluate_thr().unwrap();
        let fn_states = checkpointer.load_func_state_thr();

        let evaluator = (evaluators, fn_states).into();
        let optimizer = Op::from_state(opt_state);
        let recorder = match recorder {
            Some(mut rec) => {
                rec.after_load(&searchspace, &codomain);
                Some(rec)
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
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
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
    fn run(self) {
        let ob = Arc::new(self.objective);
        let op = Arc::new(Mutex::new(self.optimizer));
        let cod = Arc::new(self.codomain);
        let st = Arc::new(Mutex::new(self.stop));
        let scp = Arc::new(self.searchspace);
        let checkpointer = Arc::new(self.checkpointer);
        let recorder = Arc::new(self.recorder);
        let hash_evaluator = match self.evaluator {
            Some(e) => Arc::new(Mutex::new(e)),
            None => Arc::new(Mutex::new(HashFidThrSeqEvaluator::new(VecDeque::new()))),
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
            let info = Arc::new(Op::Info::default());
            let hash_evaluator = hash_evaluator.clone();

            let handle = thread::spawn(move || {
                let mut eval = match hash_evaluator.lock().unwrap().get_one_evaluator() {
                    Some(e) => e,
                    None => FidThrSeqEvaluator::new(optimizer.lock().unwrap().step(None, &scp).into(), None),
                };
                loop {
                    if stop.lock().unwrap().stop() {
                        break;
                    }
                    stop.lock().unwrap().update(ExpStep::Iteration);

                    let pair = eval.pair.take().expect("The pair ThrSeqEvaluator should not be empty (None) during evaluate.");
                    let id = pair.id();
                    let step = pair.step();
                    match step {
                        Step::Pending | Step::Partially(_) => {
                            let state = eval.state.take();
                            let fid = pair.fidelity();
                            let (out, state) = objective.compute(pair.get_sobj().get_x(), fid, state);
                            if let Some(c) = check.as_ref() {
                                c.save_func_state_thr(&id, &state);
                            }
                            
                            let new_step = out.get_step();
                            let y = cod.get_elem(&out);
                            let mut computed = pair.into_computed(y.into());
                            let outputed = (id, out);
                            computed.set_step(new_step);
                            stop.lock().unwrap().update(ExpStep::Distribution(new_step.into()));

                            if let Some(r) = rec.as_ref() {
                                r.save_pair(&computed, &outputed, &scp, &cod, Some(info.clone()));
                            }
                            if let Some(c) = check.as_ref(){
                                c.save_func_state_thr(&id, &state);
                            }
                            hash_evaluator.lock().unwrap().update_state(id, Some(state));
                            let new_sol = optimizer.lock().unwrap().step(Some(computed), &scp);
                            let new_eval = FidThrSeqEvaluator::new(new_sol, None);
                            hash_evaluator.lock().unwrap().pairs.push_front((new_eval,None));
                        },
                        _ => {
                            hash_evaluator.lock().unwrap().states.remove(&id);
                            stop.lock().unwrap().update(ExpStep::Distribution(step));
                            if let Some(c) = check.as_ref(){
                                c.remove_func_state_thr(&id);
                            }
                        },
                    }

                    eval = match hash_evaluator.lock().unwrap().get_one_evaluator() {
                        Some(e) => e,
                        None => FidThrSeqEvaluator::new(optimizer.lock().unwrap().step(None, &scp), None),
                    };
                    stop.lock().unwrap().update(ExpStep::Optimization);

                    if let Some(c) = check.as_ref() {
                        c.save_state_thr(
                            optimizer.lock().unwrap().get_state(),
                            &*stop.lock().unwrap(),
                            &eval,
                            thr,
                        );
                    }
                }
            });
            workers.push(handle);
        }
        for worker in workers {
            let _ = worker.join();
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
    Rec: DistRecorder<PSol, SId, Out, Scp, Op>,
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
    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let shapes: Vec<_> = (0..self.proc.size)
                    .map(|_| self.optimizer.step(None, &self.searchspace))
                    .collect();
                DistSeqEvaluator::new(shapes)
            }
        };
        if eval.shapes.len() < self.proc.size as usize {
            let n = self.proc.size as usize - eval.shapes.len();
            let mut shapes: Vec<_> = (0..n)
                .map(|_| self.optimizer.step(None, &self.searchspace))
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
            );
            (computed, outputed) = output.unzip();

            // Saver part
            if let Some(comp) = &computed
                && let Some(out) = &outputed
                && let Some(r) = &self.recorder
            {
                r.save_pair(comp, out, &self.searchspace, &self.codomain, None);
            }

            // Optimizer part
            eval.update(self.optimizer.step(computed, &self.searchspace));
            self.stop.update(ExpStep::Optimization);

            // Checkpoint part
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(
                    self.optimizer.get_state(),
                    &self.stop,
                    &eval,
                    self.proc.rank,
                );
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
        }
        // Receive everything overflowing stop
        sendrec.flush();
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
        FidDistSeqEvaluator<SId, Op::SInfo, Scp::SolShape>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity,
    Op: SequentialOptimizer<
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
    /// It stores a [`FuncState`] to resume previous [`Step::Partially`] evaluations.
    type WType = FidWorker<'a, SId, RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState, Check>;

    /// Create a new distributed [`MPIExperiment`] wrapped in a [`MasterWorker`] from a [`Searchspace`],
    /// [`Codomain`](crate::Codomain), [`Stepped`], [`SequentialOptimizer`], [`Stop`] condition and optional
    /// [`DistRecorder`] and [`DistCheckpointer`]. The main process (rank 0) will be the [`Master`](crate::MasterWorker) while
    /// all other processes will be [`Worker`](crate::Worker)s.
    /// It also uses an internal [`FidDistSeqEvaluator`] to evaluate single [`SolutionShape`]s per process.
    /// The [`DistRecorder`] and [`DistCheckpointer`] are only used by the main process.
    /// Other processes will use a [`WorkerCheckpointer`](crate::checkpointer::WorkerCheckpointer) associated to the [`DistCheckpointer`].
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
    /// while all other processes will be [`Worker`](crate::Worker)s loaded here via their associated [`WorkerCheckpointer`](crate::checkpointer::WorkerCheckpointer).
    /// The loading process follows the logic described in the [`DistCheckpointer`]
    /// concrete implementations (e.g. [`MessagePack`](crate::checkpointer::MessagePack)).
    ///
    /// You can use [`load!`](crate::load) macro to load an experiment more easily.
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
    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let init_vec: Vec<_> = (0..self.proc.size + 1)
                    .map(|_| self.optimizer.step(None, &self.searchspace))
                    .collect();
                FidDistSeqEvaluator::new(init_vec, self.proc.size as usize)
            }
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
            output = DistEvaluate::<_, SId, Op, Scp, Out, St, _, _, _>::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );
            (rank, solout) = output.unzip();
            (computed, outputed) = solout.unzip();

            // Saver part
            if let Some(comp) = &computed
                && let Some(out) = &outputed
                && let Some(r) = &self.recorder
            {
                r.save_pair(comp, out, &self.searchspace, &self.codomain, None);
            }

            // Optimizer part
            eval.update(self.optimizer.step(computed, &self.searchspace));
            self.stop.update(ExpStep::Optimization);
            // If more than 1 process is idle, it means that the optimizer is not able to give new solutions to at least 2 processes,
            // so the optimizer has to generate new solution ex-nihilo.
            let idle_count = sendrec.idle.count_idle();
            if idle_count > 1{
                for _ in 0..idle_count{
                    eval.update(self.optimizer.step(None, &self.searchspace));
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
        }
        // Receive everything overflowing stop
        sendrec.flush();
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
