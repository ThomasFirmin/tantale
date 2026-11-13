use std::sync::atomic::Ordering;

use crate::{
    Accumulator, BaseSol, Codomain, FidOutcome, FidPb, FidelitySol, FuncState, FuncWrapper,
    GlobalParameters, HasFidelity, HasId, HasState, HasStep, HasStepId, HasVariables, HasX,
    MultiStep, OneStep, Outcome, Problem, RUN_ID, Runable, SId, SOL_ID, Searchspace, SimplePb,
    SolInfo, SolutionShape, State, Step, StepSId,
    aliases::{Accum, Cod, CompShape, LinkObj, LinkOpt, Opt, Out, Shape},
    checkpointer::{Checkpointer, CheckpointerTargets},
    experiment::exputils::{FuncStatePool, InMemoryPool, LoadPool, Pool, PoolMode, record},
    optimizer::Sampler,
    recorder::Recorder,
    solution::shape::IntoComputedShape,
    stop::{ExpStep, Stop},
};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "Shape<P>: SolutionShape, Out<P>: Outcome")]
pub struct ExpState<P: Problem> {
    /// The current solution being evaluated.
    pub currently: Shape<P>,
}

impl<P: Problem> State for ExpState<P> {}

impl<P: Problem> ExpState<P> {
    pub fn new(currently: Shape<P>) -> Self {
        Self { currently }
    }

    pub fn to_computed(self, codomain: &Cod<P>, outcome: Out<P>) -> (CompShape<P>, Out<P>) {
        (
            self.currently
                .into_computed(codomain.get_elem(&outcome).into()),
            outcome,
        )
    }
}

impl<P: Problem> HasState for ExpState<P> {
    type State = Self;
    fn get_state(&self) -> &Self::State {
        self
    }

    fn get_mut_state(&mut self) -> &mut Self::State {
        self
    }

    fn from_state(state: Self::State) -> Self {
        state
    }
}

/// A basic experiment runner that implements the [`Runable`] trait.
/// This runner executes a single-threaded optimization loop, coordinating problem sampling,
/// evaluation, result accumulation, optional recording and checkpointing, and stop-criterion checks.
pub struct Experiment<P, O, S, R, C>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
{
    /// The [`Problem`] being solved in the experiment.
    pub problem: P,
    /// The [`Sampler`] used to generate new candidate solutions.sed to generate new candidate solutions.
    pub sampler: O,
    /// The [stopping](Stop) criterion for the experiment.
    pub stop: S,
    /// The [`Recorder`] used to log experiment data.
    pub recorder: Option<R>,
    /// The [`Checkpointer`] used to save and load experiment state.
    pub checkpointer: Option<C>,
    /// The [`Accumulator`] that collects evaluated solutions and their outcomes.
    pub accumulator: Accum<P>,
    /// The [`PoolMode`] used for managing function states during the experiment.
    pub pool_mode: PoolMode,
    /// Describes if the experiment is loaded from a previous state or started fresh.
    is_loaded: bool,
}

impl<P, O, S, R, C> Experiment<P, O, S, R, C>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
{
    pub fn new(
        problem: P,
        sampler: O,
        stop: S,
        recorder: Option<R>,
        checkpointer: Option<C>,
    ) -> Self {
        let accumulator = Accum::<P>::default();

        Self {
            problem,
            sampler,
            stop,
            recorder,
            checkpointer,
            accumulator,
            pool_mode: PoolMode::InMemory,
            is_loaded: false,
        }
    }

    pub fn with_pool(self, pool: PoolMode) -> Self {
        Self {
            pool_mode: pool,
            ..self
        }
    }

    pub fn load(problem: P, mut recorder: Option<R>, mut checkpointer: C) -> Self {
        if let Some(rec) = recorder.as_mut() {
            rec.before_load(&problem, &()).expect(
                " 
                    Recorder failed to prepare for loading the experiment state. 
                    Please ensure that the recorder is properly configured and that the necessary files are accessible.
                "
            );
        }
        checkpointer.before_load(&problem, &()).unwrap();

        let sampler = checkpointer
            .load(&CheckpointerTargets::Optimizer, &())
            .unwrap();
        let stop = checkpointer.load(&CheckpointerTargets::Stop, &()).unwrap();
        let accumulator = checkpointer
            .load(&CheckpointerTargets::Accumulator, &())
            .unwrap();
        let parameters: GlobalParameters = checkpointer
            .load(&CheckpointerTargets::Config, &())
            .unwrap();
        // if the parameters.id is greater than the current SOL_ID,
        // we replace SOl_ID by the parameters.id
        if parameters.sold_id > SOL_ID.load(Ordering::SeqCst) {
            SOL_ID.store(parameters.sold_id, Ordering::SeqCst);
        }
        RUN_ID.store(parameters.run_id + 1, Ordering::SeqCst);

        Self {
            problem,
            sampler,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            accumulator,
            pool_mode: PoolMode::InMemory,
            is_loaded: true,
        }
    }
}

impl<O, S, R, C, Scp, Out, Info> Runable for Experiment<SimplePb<Info, Scp, Out>, O, S, R, C>
where
    O: Sampler<SimplePb<Info, Scp, Out>, Scp::Opt, Info = Info>,
    S: Stop,
    R: Recorder<SimplePb<Info, Scp, Out>, ()>,
    C: Checkpointer<SimplePb<Info, Scp, Out>, ()>,
    Scp: Searchspace<OneStep<BaseSol<SId, LinkObj<Scp>, Info>, Out>> + HasVariables,
    Scp::Shape: SolutionShape<
            Id = SId,
            Info = Info,
            ObjSol = BaseSol<SId, LinkObj<Scp>, Info>,
            OptSol = BaseSol<SId, LinkOpt<Scp>, Info>,
            Obj = LinkObj<Scp>,
            Opt = LinkOpt<Scp>,
        >,
    Out: Outcome,
    Info: SolInfo,
{
    type Id = SId;
    type Context = ();

    type Problem = SimplePb<Info, Scp, Out>;
    type Sampler = O;
    type Stop = S;
    type Recorder = R;
    type Checkpointer = C;

    fn run(mut self) -> Accum<Self::Problem> {
        let mut state: ExpState<SimplePb<Info, Scp, Out>> = if self.is_loaded {
            if let Some(rec) = self.recorder.as_mut() {
                rec.after_load(&self.problem, &()).unwrap();
            }
            if let Some(check) = self.checkpointer.as_mut() {
                let exp = check.load(&CheckpointerTargets::Experiment, &());
                check.after_load(&self.problem, &()).unwrap();
                match exp {
                    Ok(e) => e,
                    Err(e) => {
                        log::warn!("
                            Failed to load experiment state: {}. Because of an error, or because
                            the experiment unintentionally stopped after an evaluation and before sampling a new solution.
                            The experiment will now attempt to recover by sampling a new solution
                            "
                            , e
                        );
                        let sol = self
                            .sampler
                            .sample(self.problem.get_searchspace(), &self.accumulator);
                        if sol.is_none() {
                            log::warn!(
                                "Optimizer returned None, stopping experiment. No new solution to evaluate."
                            );
                            return self.accumulator;
                        }
                        ExpState::new(sol.unwrap())
                    }
                }
            } else {
                panic!(
                    "The expriment was loaded from a previous state, but no checkpointer was provided."
                );
            }
        } else {
            if let Some(rec) = self.recorder.as_mut() {
                rec.init(&self.problem, &()).unwrap();
            }
            if let Some(check) = self.checkpointer.as_mut() {
                check.init(&self.problem, &()).unwrap();
            }
            let sol = self
                .sampler
                .sample(self.problem.get_searchspace(), &self.accumulator);
            if sol.is_none() {
                log::warn!(
                    "Optimizer returned None, stopping experiment. No new solution to evaluate."
                );
                return self.accumulator;
            }
            ExpState::new(sol.unwrap())
        };

        loop {
            if let Some(ref check) = self.checkpointer {
                check
                    .save_state(&state, &CheckpointerTargets::Experiment, &())
                    .map_err(|e| log::warn!("{}", e))
                    .ok();
                check
                    .save_state(&self.sampler, &CheckpointerTargets::Optimizer, &())
                    .map_err(|e| log::warn!("{}", e))
                    .ok();
                check
                    .save_state(&self.stop, &CheckpointerTargets::Stop, &())
                    .map_err(|e| log::warn!("{}", e))
                    .ok();
                check
                    .save_state(&self.accumulator, &CheckpointerTargets::Accumulator, &())
                    .map_err(|e| log::warn!("{}", e))
                    .ok();
                let parameters = GlobalParameters::new();
                check
                    .save_state(&parameters, &CheckpointerTargets::Config, &())
                    .map_err(|e| log::warn!("{}", e))
                    .ok();
            }

            // Check if stop
            if self.stop.stop() {
                break;
            };

            // Compute the output of the current solution
            let output = self
                .problem
                .get_objective()
                .compute(state.currently.get_sobj().clone_x(), ());
            let (computed, output) = state.to_computed(self.problem.get_codomain(), output);

            record(&self.recorder, &self.problem, &computed, &output);
            self.accumulator.accumulate(&computed);
            self.stop.update(ExpStep::Evaluation(Step::Evaluated));

            let new_sol = self.sampler.update_and_sample(
                &computed,
                self.problem.get_searchspace(),
                &self.accumulator,
            );
            let Some(sol) = new_sol else {
                log::warn!(
                    "Optimizer returned None, stopping experiment. No new solution to evaluate."
                );
                if let Some(ref check) = self.checkpointer {
                    check
                        .remove(&CheckpointerTargets::Experiment, &())
                        .map_err(|e| log::warn!("{}", e))
                        .ok();
                }
                break;
            };

            state = ExpState::new(sol);
            self.stop.update(ExpStep::Sampling);
            self.stop.update(ExpStep::Iteration); // New iteration
        }

        if let Some(ref check) = self.checkpointer {
            check
                .save_state(&self.sampler, &CheckpointerTargets::Optimizer, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            check
                .save_state(&self.stop, &CheckpointerTargets::Stop, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            check
                .save_state(&self.accumulator, &CheckpointerTargets::Accumulator, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            let parameters = GlobalParameters::new();
            check
                .save_state(&parameters, &CheckpointerTargets::Config, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
        }
        self.accumulator
    }
}

impl<O, S, R, C, Scp, Out, Info, FnState> Runable
    for Experiment<FidPb<Info, Scp, Out, FnState>, O, S, R, C>
where
    O: Sampler<FidPb<Info, Scp, Out, FnState>, Scp::Opt, Info = Info>,
    S: Stop,
    R: Recorder<FidPb<Info, Scp, Out, FnState>, ()>,
    C: Checkpointer<FidPb<Info, Scp, Out, FnState>, ()>,
    Scp: Searchspace<MultiStep<FidelitySol<StepSId, LinkObj<Scp>, Info>, Out, FnState>>
        + HasVariables,
    Scp::Shape: SolutionShape<
            Id = StepSId,
            Info = Info,
            ObjSol = FidelitySol<StepSId, LinkObj<Scp>, Info>,
            OptSol = FidelitySol<StepSId, LinkOpt<Scp>, Info>,
            Obj = LinkObj<Scp>,
            Opt = LinkOpt<Scp>,
        > + HasFidelity
        + HasStep,
    CompShape<FidPb<Info, Scp, Out, FnState>>: HasStep + HasStepId,
    Out: FidOutcome,
    Info: SolInfo,
    FnState: FuncState,
{
    type Id = StepSId;
    type Context = ();

    type Problem = FidPb<Info, Scp, Out, FnState>;
    type Sampler = O;
    type Stop = S;
    type Recorder = R;
    type Checkpointer = C;

    fn run(mut self) -> Accum<Self::Problem> {
        let mut state: ExpState<FidPb<Info, Scp, Out, FnState>> = if self.is_loaded {
            if let Some(rec) = self.recorder.as_mut() {
                rec.after_load(&self.problem, &()).unwrap();
            }
            if let Some(check) = self.checkpointer.as_mut() {
                let exp = check.load(&CheckpointerTargets::Experiment, &());
                check.after_load(&self.problem, &()).unwrap();
                match exp {
                    Ok(e) => e,
                    Err(e) => {
                        log::warn!("
                            Failed to load experiment state: {}. Because of an error, or because
                            the experiment unintentionally stopped after an evaluation and before sampling a new solution.
                            The experiment will now attempt to recover by sampling a new solution
                            "
                            , e
                        );
                        let sol = self
                            .sampler
                            .sample(self.problem.get_searchspace(), &self.accumulator);
                        if sol.is_none() {
                            log::warn!(
                                "Optimizer returned None, stopping experiment. No new solution to evaluate."
                            );
                            return self.accumulator;
                        }
                        ExpState::new(sol.unwrap())
                    }
                }
            } else {
                panic!(
                    "The expriment was loaded from a previous state, but no checkpointer was provided."
                );
            }
        } else {
            if let Some(rec) = self.recorder.as_mut() {
                rec.init(&self.problem, &()).unwrap();
            }
            if let Some(check) = self.checkpointer.as_mut() {
                check.init(&self.problem, &()).unwrap();
            }
            let sol = self
                .sampler
                .sample(self.problem.get_searchspace(), &self.accumulator);
            if sol.is_none() {
                log::warn!(
                    "Optimizer returned None, stopping experiment. No new solution to evaluate."
                );
                return self.accumulator;
            }
            ExpState::new(sol.unwrap())
        };

        let mut pool = match self.pool_mode {
            PoolMode::InMemory => Pool::InMemory(InMemoryPool::default()),
            PoolMode::Persistent => Pool::Persistent(LoadPool::default()),
        };

        loop {
            if let Some(ref check) = self.checkpointer {
                check
                    .save_state(&state, &CheckpointerTargets::Experiment, &())
                    .map_err(|e| log::warn!("{}", e))
                    .ok();
                check
                    .save_state(&self.sampler, &CheckpointerTargets::Optimizer, &())
                    .map_err(|e| log::warn!("{}", e))
                    .ok();
                check
                    .save_state(&self.stop, &CheckpointerTargets::Stop, &())
                    .map_err(|e| log::warn!("{}", e))
                    .ok();
                check
                    .save_state(&self.accumulator, &CheckpointerTargets::Accumulator, &())
                    .map_err(|e| log::warn!("{}", e))
                    .ok();
                let parameters = GlobalParameters::new();
                check
                    .save_state(&parameters, &CheckpointerTargets::Config, &())
                    .map_err(|e| log::warn!("{}", e))
                    .ok();
            }

            // Check if stop
            if self.stop.stop() {
                break;
            };

            let result = match state.currently.step() {
                Step::Pending => {
                    let fidelity = state.currently.fidelity();
                    Some(
                        self.problem
                            .get_objective()
                            .compute(state.currently.get_sobj().clone_x(), (fidelity, None)),
                    )
                }
                Step::Partially(_) => {
                    // Compute the output of the current solution
                    let fn_state = pool.retrieve(state.currently.ref_id(), &self.checkpointer, &());
                    let fidelity = state.currently.fidelity();
                    Some(
                        self.problem
                            .get_objective()
                            .compute(state.currently.get_sobj().clone_x(), (fidelity, fn_state)),
                    )
                }
                Step::Discard => {
                    pool.remove(state.currently.ref_id(), &self.checkpointer, &());
                    self.stop.update(ExpStep::Evaluation(Step::Discard));
                    None
                }
                Step::Evaluated => panic!(
                    "The current solution is already evaluated. This should not happen in the experiment loop."
                ),
            };

            if let Some((output, fnstate)) = result {
                let (mut computed, output) = state.to_computed(self.problem.get_codomain(), output);

                pool.remove(computed.ref_id(), &self.checkpointer, &());
                computed.increment();
                computed.set_raw_step(output.get_step());
                if matches!(computed.step(), Step::Partially(_)) {
                    pool.insert(computed.id(), fnstate, &self.checkpointer, &());
                }

                record(&self.recorder, &self.problem, &computed, &output);
                self.accumulator.accumulate(&computed);
                self.stop.update(ExpStep::Evaluation(computed.step()));
                self.sampler
                    .update(&computed, self.problem.get_searchspace(), &self.accumulator);
            }

            let new_sol = self
                .sampler
                .sample(self.problem.get_searchspace(), &self.accumulator);
            let Some(sol) = new_sol else {
                log::warn!(
                    "Optimizer returned None, stopping experiment. No new solution to evaluate."
                );
                if let Some(ref check) = self.checkpointer {
                    check
                        .remove(&CheckpointerTargets::Experiment, &())
                        .map_err(|e| log::warn!("{}", e))
                        .ok();
                }
                break;
            };

            state = ExpState::new(sol);
            self.stop.update(ExpStep::Sampling);
            self.stop.update(ExpStep::Iteration); // New iteration
        }

        if let Some(ref check) = self.checkpointer {
            check
                .save_state(&self.sampler, &CheckpointerTargets::Optimizer, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            check
                .save_state(&self.stop, &CheckpointerTargets::Stop, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            check
                .save_state(&self.accumulator, &CheckpointerTargets::Accumulator, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            let parameters = GlobalParameters::new();
            check
                .save_state(&parameters, &CheckpointerTargets::Config, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
        }
        self.accumulator
    }
}
