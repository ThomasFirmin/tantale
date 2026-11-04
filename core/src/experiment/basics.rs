use std::{collections::HashMap, marker::PhantomData, ops::Deref, process::exit, sync::{Arc, Mutex, atomic::Ordering}, thread, time::Duration};

use crate::{
    Accumulator, BaseSol, Checkpointer, Codomain, Computed, FidOutcome, FidPb, FidelitySol, FuncState, FuncWrapper, GlobalParameters, HasFidelity, HasId, HasState, HasStep, HasStepId, HasVariables, HasX, MultiStep, OneStep, Outcome, Problem, RUN_ID, Runable, SId, SOL_ID, Sampler, Searchspace, SimplePb, SolId, SolInfo, SolutionShape, State, Step, StepId, StepSId, Stop, Var, aliases::{Accum, Cod, CompShape, LinkObj, LinkOpt, Obj, Opt, Out, SObj, SOpt, Shape, TrueOpt, XObj}, checkpointer::CheckpointerTargets, errors::{CommunicatorError, WorkerError}, experiment::{
        PoolMode, parallel::{MasterWorkerCom, WorkerHandle, WorkerTrait}}, objective::outcome::IdOut, recorder::{Recorder, RecorderTargets}, solution::shape::IntoComputedShape, stop::ExpStep, utils::{Communicator, communicator::{AskMsg, Channel, Envelope, InitMsg, Msg, OutMsg, StopMsg, XMsg}, record::{CompParts, Recordable, Schema, SolParts}
    }
};

use indexmap::IndexMap;
use log::warn;
use serde::{Deserialize, Serialize};
use tantale_macros::basic_bounds;

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "Shape<P>: SolutionShape, Out<P>: Outcome")]
pub struct ExpState<P:Problem> {
    /// The current solution being evaluated.
    pub currently: Shape<P>,
    /// The current output of the optimization process, if any.
    pub output: Option<Out<P>>,
}

impl<P:Problem> State for ExpState<P> { }

impl<P:Problem> ExpState<P> {
    pub fn new(currently: Shape<P>) -> Self {
        Self { currently, output: None }
    }

    pub fn set_output(&mut self, output: Out<P>) {
        self.output = Some(output);
    }

    pub fn has_output(&self) -> bool {
        self.output.is_some()
    }

    pub fn to_computed(self, codomain: &Cod<P>) -> (CompShape<P>, Out<P>) {
        let output = self.output.expect("No output available to convert to computed shape.");
        (self.currently.into_computed(codomain.get_elem(&output).into()), output)
    }
}

impl<P:Problem> HasState for ExpState<P> {
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
    /// The [`Accumulator`] that collects best evaluated solutions and their outcomes.
    pub accumulator: Accum<P>,
    /// The [`PoolMode`] used for managing function states during the experiment.
    pub pool_mode: PoolMode,
    /// The current state of the experiment.
    state: ExpState<P>,
}

impl<P, O, S, R, C> Experiment<P, O, S, R, C>
where
    P:Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S:Stop,
    R:Recorder<P, ()>,
    C:Checkpointer<P, ()>,
{

    pub fn new(
        problem: P,
        mut sampler: O,
        stop: S,
        mut recorder: Option<R>,
        mut checkpointer: Option<C>,
    ) -> Self 
    {
        let accumulator = Accum::<P>::default();
        let sol = sampler.sample(problem.get_searchspace(), &accumulator);
        if sol.is_none() {
            log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate.");
            exit(1);
        }
        let state = ExpState::new(sol.unwrap());

        if let Some(rec) = recorder.as_mut(){
            rec.init(&problem, &()).unwrap();
        }

        if let Some(check) = checkpointer.as_mut() {
            check.init(&problem, &()).unwrap();
        }

        Self {
            problem,
            sampler,
            stop,
            recorder,
            checkpointer,
            accumulator,
            pool_mode: PoolMode::InMemory,
            state,
        }
    }

    pub fn with_pool(self, pool: PoolMode) -> Self {
        Self {
            pool_mode: pool,
            ..self
        }
    }

    pub fn load(problem: P, mut recorder: Option<R>, mut checkpointer: C) -> Self
    {
        if let Some(rec) = recorder.as_mut() {
            rec.before_load(&problem, &()).expect(
                " 
                    Recorder failed to prepare for loading the experiment state. 
                    Please ensure that the recorder is properly configured and that the necessary files are accessible.
                "
            );
        }
        checkpointer.before_load(&problem, &()).unwrap();

        let mut sampler: O = checkpointer.load(&CheckpointerTargets::Optimizer, &()).unwrap();
        let stop = checkpointer.load(&CheckpointerTargets::Stop, &()).unwrap();
        let accumulator = checkpointer.load(&CheckpointerTargets::Accumulator, &()).unwrap();
        let parameters: GlobalParameters = checkpointer.load(&CheckpointerTargets::Config, &()).unwrap();
        // if the parameters.id is greater than the current SOL_ID, 
        // we replace SOl_ID by the parameters.id 
        if parameters.sold_id > SOL_ID.load(Ordering::SeqCst) {
            SOL_ID.store(parameters.sold_id, Ordering::SeqCst);
        }
        RUN_ID.store(parameters.run_id + 1, Ordering::SeqCst);
        let exp = checkpointer.load(&CheckpointerTargets::Experiment, &());

        if let Some(rec) = recorder.as_mut() {
            rec.after_load(&problem, &()).unwrap();
        }
        checkpointer.after_load(&problem, &()).unwrap();


        let state = match exp {
            Ok(e) => e,
            Err(e) => {
                log::warn!("
                    Failed to load experiment state: {}. Because of an error, or because
                    the experiment unintentionally stopped after an evaluation and before sampling a new solution.
                    The experiment will now attempt to recover by sampling a new solution
                    "
                    , e
                );
                let sol = sampler.sample(problem.get_searchspace(), &accumulator);
                if sol.is_none() {
                    log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate.");
                    exit(1);
                }
                ExpState::new(sol.unwrap())
            },
        };

        Self {
            problem,
            sampler,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            accumulator,
            pool_mode: PoolMode::InMemory,
            state,
        }
    }
}

fn record<R: Recorder<P,Ctx>, P: Problem, Ctx>
(
    recorder: &Option<R>,
    problem: &P,
    computed: &CompShape<P>,
    outputed: &Out<P>,
)
where
    P::Id: SolId,
    P::Info: SolInfo,
    P::Scp: HasVariables,
    P::Func: FuncWrapper<Dom = LinkObj<P::Scp>>,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts> + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>] ,&'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
{
    if let Some(rec) = recorder {
        let codomain = problem.get_codomain();
        let vars = problem.get_searchspace().get_vars();
        let contexts = (codomain, &(vars, codomain));
        rec.record(&RecorderTargets::XObjective, computed.get_sobj(), contexts, CompParts::Sol(SolParts::X)).unwrap();
        rec.record(&RecorderTargets::XOptimizer, computed.get_sopt(), contexts, CompParts::Sol(SolParts::X)).unwrap();
        rec.record(&RecorderTargets::Y, computed.get_sobj(), contexts, CompParts::Y).unwrap();
        rec.record(&RecorderTargets::Outcome, &IdOut(computed.ref_id(),outputed), (&(),&()), ()).unwrap();
        rec.record(&RecorderTargets::Metadata, computed.get_sobj(), contexts, CompParts::Sol(SolParts::Info)).unwrap();
    }
}

impl<O, S, R, C, Scp, Out, Info> Runable for Experiment<SimplePb<Info, Scp, Out>, O, S, R, C>
where
    O:Sampler<SimplePb<Info, Scp, Out>, Scp::Opt, Info = Info>,
    S:Stop,
    R:Recorder<SimplePb<Info, Scp, Out>, ()>,
    C:Checkpointer<SimplePb<Info, Scp, Out>, ()>,
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
    type Id  = SId;
    type Context = ();

    type Problem = SimplePb<Info, Scp, Out>;
    type Sampler = O;
    type Stop = S;
    type Recorder = R;
    type Checkpointer = C;
    
    fn run(mut self) -> Accum<Self::Problem> 
    {
        if self.state.has_output() 
        {
            // If the experiment state has an output, it means that the last evlauation was recorded
            // but not yet recorded, and accumulated. So we need to record it and update the accumulator.
            let (computed, output) = self.state.to_computed(self.problem.get_codomain());
            record(&self.recorder, &self.problem, &computed, &output);

            self.accumulator.accumulate(&computed);
            let new_sol = self.sampler.update_and_sample(&computed, self.problem.get_searchspace(), &self.accumulator);
            let Some(sol) = new_sol else { log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate."); return self.accumulator; };
            self.state = ExpState::new(sol);
            self.stop.update(ExpStep::Sampling);
        }
        
        loop {
            // Check if stop
            if let Some(ref check) = self.checkpointer {
                check.save_state(&self.state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.sampler, &CheckpointerTargets::Optimizer, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.stop, &CheckpointerTargets::Stop, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.accumulator, &CheckpointerTargets::Accumulator, &()).map_err(|e| log::warn!("{}", e)).ok();
                let parameters = GlobalParameters::new();
                check.save_state(&parameters, &CheckpointerTargets::Config, &()).map_err(|e| log::warn!("{}", e)).ok();
            }
            if self.stop.stop() { break; };

            // Compute the output of the current solution
            let output = self.problem.get_objective().compute(self.state.currently.get_sobj().clone_x(), ());
            
            self.state.set_output(output);
            if let Some(ref check) = self.checkpointer {
                check.save_state(&self.state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
            }
            
            let (computed, output) = self.state.to_computed(self.problem.get_codomain());

            record(&self.recorder, &self.problem, &computed, &output);
            self.accumulator.accumulate(&computed);

            if let Some(ref check) = self.checkpointer {
                check.remove(&CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
            }
            self.stop.update(ExpStep::Evaluation(Step::Evaluated));

            let new_sol = self.sampler.update_and_sample(&computed, self.problem.get_searchspace(), &self.accumulator);
            let Some(sol) = new_sol else { log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate."); return self.accumulator; };

            self.state = ExpState::new(sol);
            self.stop.update(ExpStep::Sampling);
            self.stop.update(ExpStep::Iteration); // New iteration
        }

        self.accumulator
    }
    
    fn get_stop(&self) -> &Self::Stop {
        &self.stop
    }
    
    fn searchspace(&self) -> &crate::aliases::Scp<Self::Problem> {
        self.problem.get_searchspace()
    }
    
    fn codomain(&self) -> &Cod<Self::Problem> {
        self.problem.get_codomain()
    }
    
    fn objective(&self) -> &crate::aliases::Func<Self::Problem> {
        self.problem.get_objective()
    }
    
    fn sampler(&self) -> &Self::Sampler {
        &self.sampler
    }
    
    fn recorder(&self) -> Option<&Self::Recorder> {
        self.recorder.as_ref()
    }
    
    fn checkpointer(&self) -> Option<&Self::Checkpointer> {
        self.checkpointer.as_ref()
    }

    fn accumulator(&self) -> &Accum<Self::Problem> {
        &self.accumulator
    }
    
    fn mut_stop(&mut self) -> &mut Self::Stop {
        &mut self.stop
    }
    
    fn mut_searchspace(&mut self) -> &mut crate::aliases::Scp<Self::Problem> {
        self.problem.get_mut_searchspace()
    }
    
    fn mut_codomain(&mut self) -> &mut Cod<Self::Problem> {
        self.problem.get_mut_codomain()
    }
    
    fn mut_objective(&mut self) -> &mut crate::aliases::Func<Self::Problem> {
        self.problem.get_mut_objective()
    }
    
    fn mut_sampler(&mut self) -> &mut Self::Sampler {
        &mut self.sampler
    }
    
    fn mut_recorder(&mut self) -> Option<&mut Self::Recorder> {
        self.recorder.as_mut()
    }
    
    fn mut_checkpointer(&mut self) -> Option<&mut Self::Checkpointer> {
        self.checkpointer.as_mut()
    }
    
    fn mut_accumulator(&mut self) -> &mut Accum<Self::Problem> {
        &mut self.accumulator
    }
}

/// Trait defining the interface for a pool of [`FuncState`]s associated with solution identifiers.
/// This abstraction allows for different implementations of function state management, such as in-memory pools or checkpointer-backed storage.
pub trait FuncStatePool<Id: StepId, FnState: FuncState>: Default
{
    /// Insert a new [`FuncState`] into the pool, associated with a solution [`StepId`].
    fn insert<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: Id, state: FnState, check: &Option<C>,ctx: &Ctx);
    /// Remove the [`FuncState`] associated with the given solution [`StepId`], if it exists.
    fn remove<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> bool;
    /// Retrieve the [`FuncState`] associated with the given solution [`StepId`], if it exists.
    fn retrieve<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> Option<FnState>;
}

/// An implementation of [`FuncStatePool`] that uses an in-memory [`IndexMap`] to store function states.
/// This pool is suitable for scenarios where memory is sufficient to hold all function states.
pub struct InMemoryPool<Id: StepId, FnState: FuncState>(IndexMap<Id, FnState>);

impl<Id: StepId, FnState: FuncState> InMemoryPool<Id, FnState>
{
    /// Create a new [`InMemoryPool`] with an optional function state checkpointer.
    pub fn new() -> Self {
        Self(IndexMap::new())
    }
}

impl<Id: StepId, FnState: FuncState> Default for InMemoryPool<Id, FnState>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Id: StepId, FnState: FuncState> FromIterator<(Id, FnState)> for InMemoryPool<Id, FnState>
{
    fn from_iter<T: IntoIterator<Item = (Id, FnState)>>(iter: T) -> Self {
        let pool = iter.into_iter().collect();
        Self(pool)
    }
}

impl<Id: StepId, FnState: FuncState> FuncStatePool<Id, FnState> for InMemoryPool<Id, FnState>
{
    /// Insert a new [`FuncState`] into the pool, associated with the given solution [`StepId`].
    fn insert<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: Id, state: FnState, check: &Option<C>,ctx: &Ctx) {
        if let Some(c) = check {
            let _ = c.save_fn_state(&id, &state, ctx);
        }
        self.0.insert(id, state);
    }
    /// Remove the [`FuncState`] associated with the given solution [`StepId`], if it exists.
    /// Returns `true` if the state was present and removed, `false` otherwise.
    fn remove<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> bool {
        if let Some(c) = check {
            let _ = c.remove_all_previous_fn_states::<Id, FnState>(id, ctx);
        }
        self.0.swap_remove(id).is_some()
    }
    /// Retrieve the [`FuncState`] associated with the given solution [`StepId`], if it exists.
    fn retrieve<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> Option<FnState> {
        let res = self.0.swap_remove(id);
        if let Some(c) = check && res.is_none() {
            let loaded = c.load_fn_state::<Id, FnState>(*id, ctx);
            loaded.map_or(None, |a| Some(a.1))
        } else {
            res
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadPool<Id: StepId, FnState: FuncState>(PhantomData<(Id, FnState)>);

impl<Id: StepId, FnState: FuncState> LoadPool<Id, FnState>
{
    /// Create a new [`LoadPool`] with an optional function state checkpointer.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Id: StepId, FnState: FuncState> Default for LoadPool<Id, FnState>
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Id: StepId, FnState: FuncState> FuncStatePool<Id, FnState> for LoadPool<Id, FnState>
{
    /// Insert a new [`FuncState`] into the pool, associated with the given solution [`StepId`].
    /// This implementation saves the state using the checkpointer and keeps track of the current state in memory for quick retrieval.
    fn insert<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: Id, state: FnState, check: &Option<C>,ctx: &Ctx) {
        if let Some(c) = check {
            let _ = c.save_fn_state(&id, &state, ctx);
        } else {
            panic!("LoadPool requires a checkpointer to save function states.");
        }
    }

    /// Remove the [`FuncState`] associated with the given solution [`StepId`], if it exists.
    /// This implementation removes the state from the checkpointer and clears the current state if it matches the given [`StepId`].
    fn remove<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> bool {
        if let Some(c) = check {
            let _ = c.remove_all_previous_fn_states::<Id, FnState>(id, ctx);
            true
        } else {
            panic!("LoadPool requires a checkpointer to remove function states.");
        }
    }

    /// Retrieve the [`FuncState`] associated with the given solution [`StepId`], if it exists.
    /// This implementation first checks if the current state in memory matches the given [`StepId`] and
    /// returns it if so; otherwise, it attempts to load the state from the checkpointer.
    fn retrieve<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> Option<FnState> {
        if let Some(c) = check {
            let loaded = c.load_fn_state::<Id, FnState>(*id, ctx);
            loaded.map_or(None, |a| Some(a.1))
        } else {
            panic!("LoadPool requires a checkpointer to retrieve function states.");
        }
    }
}

pub enum Pool<Id: StepId, FnState: FuncState>
{
    InMemory(InMemoryPool<Id, FnState>),
    Persistent(LoadPool<Id, FnState>),
}


impl<Id: StepId, FnState: FuncState> Pool<Id, FnState> {
    pub fn new(mode: PoolMode) -> Self {
        match mode {
            PoolMode::InMemory => Self::InMemory(InMemoryPool::default()),
            PoolMode::Persistent => Self::Persistent(LoadPool::default()),
        }
    }
}

impl<Id: StepId, FnState: FuncState> Default for Pool<Id, FnState> {
    fn default() -> Self {
        Self::InMemory(InMemoryPool::default())
    }
}

impl<Id: StepId, FnState: FuncState> FuncStatePool<Id, FnState> for Pool<Id, FnState> {
    fn insert<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: Id, state: FnState, check: &Option<C>,ctx: &Ctx) {
        match self {
            Pool::InMemory(p) => p.insert(id, state, check, ctx),
            Pool::Persistent(p) => p.insert(id, state, check, ctx),
        }
    }

    fn remove<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> bool {
        match self {
            Pool::InMemory(p) => p.remove(id, check, ctx),
            Pool::Persistent(p) => p.remove(id, check, ctx),
        }
    }

    fn retrieve<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> Option<FnState> {
        match self {
            Pool::InMemory(p) => p.retrieve(id, check, ctx),
            Pool::Persistent(p) => p.retrieve(id, check, ctx),
        }
    }
}


impl<O, S, R, C, Scp, Out, Info, FnState> Runable for Experiment<FidPb<Info, Scp, Out, FnState>, O, S, R, C>
where
    O:Sampler<FidPb<Info, Scp, Out, FnState>, Scp::Opt, Info = Info>,
    S:Stop,
    R:Recorder<FidPb<Info, Scp, Out, FnState>, ()>,
    C:Checkpointer<FidPb<Info, Scp, Out, FnState>, ()>,
    Scp: Searchspace<MultiStep<FidelitySol<StepSId, LinkObj<Scp>, Info>, Out, FnState>> + HasVariables,
    Scp::Shape: SolutionShape<
        Id = StepSId,
        Info = Info,
        ObjSol = FidelitySol<StepSId, LinkObj<Scp>, Info>,
        OptSol = FidelitySol<StepSId, LinkOpt<Scp>, Info>,
        Obj = LinkObj<Scp>,
        Opt = LinkOpt<Scp>,
    > + HasFidelity + HasStep,
    CompShape<FidPb<Info, Scp, Out, FnState>>: HasStep + HasStepId,
    Out: FidOutcome,
    Info: SolInfo,
    FnState: FuncState,
{
    type Id  = StepSId;
    type Context = ();

    type Problem = FidPb<Info, Scp, Out, FnState>;
    type Sampler = O;
    type Stop = S;
    type Recorder = R;
    type Checkpointer = C;
    
    fn run(mut self) -> Accum<Self::Problem> 
    {
        if self.state.has_output() 
        {
            // If the experiment state has an output, it means that the last evlauation was recorded
            // but not yet recorded, and accumulated. So we need to record it and update the accumulator.
            let (computed, output) = self.state.to_computed(self.problem.get_codomain());
            record(&self.recorder, &self.problem, &computed, &output);

            self.accumulator.accumulate(&computed);
            let new_sol = self.sampler.update_and_sample(&computed, self.problem.get_searchspace(), &self.accumulator);
            let Some(sol) = new_sol else { log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate."); return self.accumulator; };
            self.state = ExpState::new(sol);
            self.stop.update(ExpStep::Sampling);
        }

        let mut pool = match self.pool_mode {
            PoolMode::InMemory => Pool::InMemory(InMemoryPool::default()),
            PoolMode::Persistent => Pool::Persistent(LoadPool::default()),
        };
        
        loop {
            if let Some(ref check) = self.checkpointer {
                check.save_state(&self.state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.sampler, &CheckpointerTargets::Optimizer, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.stop, &CheckpointerTargets::Stop, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.accumulator, &CheckpointerTargets::Accumulator, &()).map_err(|e| log::warn!("{}", e)).ok();
                let parameters = GlobalParameters::new();
                check.save_state(&parameters, &CheckpointerTargets::Config, &()).map_err(|e| log::warn!("{}", e)).ok();
            }
            // Check if stop
            if self.stop.stop() { break; };
            
            let result = match self.state.currently.step() {
                Step::Pending => {
                    let fidelity = self.state.currently.fidelity();
                    Some(
                        self.problem.get_objective().compute(self.state.currently.get_sobj().clone_x(), (fidelity, None))
                    )
                },
                Step::Partially(_) => {
                    // Compute the output of the current solution
                    let fn_state = pool.retrieve(self.state.currently.ref_id(), &self.checkpointer, &());
                    let fidelity = self.state.currently.fidelity();
                    Some(
                        self.problem.get_objective().compute(self.state.currently.get_sobj().clone_x(), (fidelity, fn_state))
                    )
                },
                Step::Discard => {
                    pool.remove(self.state.currently.ref_id(), &self.checkpointer, &());
                    None
                },
                Step::Evaluated => {
                    warn!("Solution with ID: {:?} has already been evaluated. Skipping evaluation.", self.state.currently.ref_id());
                    pool.remove(self.state.currently.ref_id(), &self.checkpointer, &());
                    None
                },
                Step::Error => {
                    warn!("Solution with ID: {:?} encountered an error during evaluation. Skipping evaluation.",
                    self.state.currently.ref_id());
                    pool.remove(self.state.currently.ref_id(), &self.checkpointer, &());
                    None
                },
            };
            
            let computed = if let Some((output, state)) = result{
                self.state.set_output(output);        
                if let Some(ref check) = self.checkpointer {
                    check.save_state(&self.state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
                }
                pool.remove(self.state.currently.ref_id(), &self.checkpointer, &());

                let (mut computed, output) = self.state.to_computed(self.problem.get_codomain());
                computed.increment();
                computed.set_raw_step(output.get_step());
    
                if matches!(computed.step(), Step::Partially(_)) {
                    pool.insert(computed.id(), state, &self.checkpointer, &());
                }
                record(&self.recorder, &self.problem, &computed, &output);
                self.accumulator.accumulate(&computed);
                self.stop.update(ExpStep::Evaluation(computed.step()));
                Some(computed)
            } else {
                None
            };

            if let Some(ref check) = self.checkpointer {
                check.remove(&CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
            }

            if let Some(comp) = &computed {
                self.sampler.update(comp, self.problem.get_searchspace(), &self.accumulator);
            }
            let new_sol = self.sampler.sample(self.problem.get_searchspace(), &self.accumulator);
            let Some(sol) = new_sol else { log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate."); return self.accumulator; };

            self.state = ExpState::new(sol);
            self.stop.update(ExpStep::Sampling);
            self.stop.update(ExpStep::Iteration); // New iteration
        }

        self.accumulator
    }
    
    fn get_stop(&self) -> &Self::Stop {
        &self.stop
    }
    
    fn searchspace(&self) -> &crate::aliases::Scp<Self::Problem> {
        self.problem.get_searchspace()
    }
    
    fn codomain(&self) -> &Cod<Self::Problem> {
        self.problem.get_codomain()
    }
    
    fn objective(&self) -> &crate::aliases::Func<Self::Problem> {
        self.problem.get_objective()
    }
    
    fn sampler(&self) -> &Self::Sampler {
        &self.sampler
    }
    
    fn recorder(&self) -> Option<&Self::Recorder> {
        self.recorder.as_ref()
    }
    
    fn checkpointer(&self) -> Option<&Self::Checkpointer> {
        self.checkpointer.as_ref()
    }

    fn accumulator(&self) -> &Accum<Self::Problem> {
        &self.accumulator
    }
    
    fn mut_stop(&mut self) -> &mut Self::Stop {
        &mut self.stop
    }
    
    fn mut_searchspace(&mut self) -> &mut crate::aliases::Scp<Self::Problem> {
        self.problem.get_mut_searchspace()
    }
    
    fn mut_codomain(&mut self) -> &mut Cod<Self::Problem> {
        self.problem.get_mut_codomain()
    }
    
    fn mut_objective(&mut self) -> &mut crate::aliases::Func<Self::Problem> {
        self.problem.get_mut_objective()
    }
    
    fn mut_sampler(&mut self) -> &mut Self::Sampler {
        &mut self.sampler
    }
    
    fn mut_recorder(&mut self) -> Option<&mut Self::Recorder> {
        self.recorder.as_mut()
    }
    
    fn mut_checkpointer(&mut self) -> Option<&mut Self::Checkpointer> {
        self.checkpointer.as_mut()
    }
    
    fn mut_accumulator(&mut self) -> &mut Accum<Self::Problem> {
        &mut self.accumulator
    }
}


//---------------------//
// Parallel Experiment //
//---------------------//


#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "Shape<P>: SolutionShape, Out<P>: Outcome")]
pub struct ParState<P:Problem> {
    /// The current solution being evaluated.
    pub currently: Vec<Option<Shape<P>>>,
}

impl<P:Problem> State for ParState<P> { }

impl<P:Problem> ParState<P> {
    pub fn new(size: usize) -> Self {
        Self { currently: vec![None; size] }
    }

    pub fn set_currently(&mut self, index: usize, currently: Shape<P>) {
        if index < self.currently.len() {
            self.currently[index] = Some(currently);
        } else {
            log::warn!("Index {} is out of bounds for currently vector. No update performed.", index);
        }
    }
}

impl<P:Problem> HasState for ParState<P> {
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

fn receive_with_tolerance<Chan>(chan: &Chan, tolerance: usize) -> Result<Envelope, String>
where
    Chan: Channel,
{
    let mut receive_error_count = 0;
    loop {
        let msg = chan.receive();
        match msg {
            Err(e) => {
                receive_error_count += 1;
                if receive_error_count > tolerance {
                    return Err(format!("Failed to receive a message after {} blocking attempts: {:?}", receive_error_count, e));
                }
            },
            Ok(envelope) => {
                return Ok(envelope);
            }
        }
    }
}

fn try_receive_with_tolerance<Chan>(chan: &Chan, tolerance: usize) -> Result<Option<Envelope>, String>
where
    Chan: Channel,
{
    let mut receive_error_count = 0;
    loop {
        let msg = chan.try_receive();
        match msg {
            Err(e) => {
                receive_error_count += 1;
                if receive_error_count > tolerance {
                    return Err(format!("Failed to receive a message after {} blocking attempts: {:?}", receive_error_count, e));
                }
            },
            Ok(envelope) => {
                return Ok(envelope);
            }
        }
    }
}

#[basic_bounds(P)]
#[allow(clippy::too_many_arguments)]
fn receive_and_update<O, S, R, P, Com>(
    com: &mut Com,
    problem: &P,
    accumulator: &mut Accum<P>,
    sampler: &mut O,
    stop: &mut S,
    recorder: &mut Option<R>,
    state: &mut ParState<P>,
    tolerance: usize,
) -> Result<(), String>
where
    P: 'static,
    O:Sampler<P, Opt<P>, Info = P::Info>,
    S:Stop,
    R:Recorder<P, ()>,
    Com: MasterWorkerCom,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts> + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>] ,&'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
{
    // receive all available messages from workers, with a tolerance for blocking attempts
    loop {
        if stop.stop() {
            break;
        }
        let mut msg = try_receive_with_tolerance(com.channel(0), tolerance);
        match msg {
            Err(e) => return Err(e),
            Ok(ref mut envelope) => {
                match envelope.take() {
                    None => break, // No more messages to process
                    Some(envelope) => {
                        let tag = envelope.tag;
                        let from = envelope.from;
                        com.set_idle(from);
                        if com.is_dead(from){
                            log::warn!("The master received a message from a dead Worker {}. This worker is now considered as alive", from);
                            com.set_alive(from);
                        }
                        match tag {
                            InitMsg::TAG => { // Initialize with a potential loaded solution
                                log::info!("Worker {} initialized and ready to receive tasks.", from);
                                com.set_alive(from);
                                let res = fill_idle_with_state(com, state, from);
                                if let Err(e) = res {
                                    log::warn!(
                                        "
                                            Failed to send a solution to worker {}.
                                            The worker will be considered as dead.
                                            Error: {:?}
                                        ", from, &e
                                    );
                                    com.set_dead(from);
                                }
                            },
                            OutMsg::<P>::TAG => { // Usual f(x) = y management
                                let msg = envelope.content.into_any().downcast::<OutMsg<P>>().unwrap();
                                let out = &msg.as_ref().1;
                                let sol = state.currently[from].take();
                                match sol {
                                    Some(s) => {
                                        let y = problem.get_codomain().get_elem(out).into();
                                        let computed = s.into_computed(y);
                                        record(recorder, problem, &computed, out);
                                        stop.update(ExpStep::Evaluation(Step::Evaluated));
                                        accumulator.accumulate(&computed);
                                        sampler.update(&computed, problem.get_searchspace(), accumulator);
                                    },
                                    None => {
                                        log::warn!("
                                            Received output for solution ID {:?} from worker {}, 
                                            but no corresponding solution was found in the current state. 
                                            This may indicate a mismatch between sent and received solutions.
                                            Or failure when loading a previous state from a checkpoint.
                                            ",
                                            msg.as_ref().0,
                                            from
                                        );
                                    }
                                }
                            },
                            _ => {
                                log::warn!("Received unexpected message: {:?}. Worker will be considered as idle", tag);
                                // Do nothing, just mark the worker as idle
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[basic_bounds(P)]
fn fill_idle_with_state<P, Com>(
    com: &mut Com,
    state: &mut ParState<P>,
    peer:usize,
) -> Result<(), CommunicatorError>
where
    P: 'static,
    Com: MasterWorkerCom,
{
    if com.is_busy(peer) || com.is_dead(peer) {
        return Err(CommunicatorError::SendError("The peer the master process tries to communicate with, is busy or idle.".to_string()));
    }
    if let Some(sol) = &state.currently[peer]{
        let x = sol.get_sobj().clone_x();
        let id = sol.id();
        let msg = XMsg::<P>(id, x);
        let envelope = Envelope::new(0, peer, msg);
        let res = com.channel(0).send(envelope);
        com.set_busy(peer);
        res
    } else {
        Ok(())
    }
}

#[basic_bounds(P)]
fn fill_all_idles<O, S, P, Com>(
    com: &mut Com,
    sampler: &mut O,
    scp: &P::Scp,
    accumulator: &mut Accum<P>,
    stop: &mut S,
    state: &mut ParState<P>,
) -> Result<(), String>
where
    P: 'static,
    O:Sampler<P, Opt<P>, Info = P::Info>,
    S:Stop,
    Com: MasterWorkerCom,
{
    while let Some(peer) = com.get_idle() 
        && let Some(sol) = sampler.sample(scp, accumulator) 
        && !stop.stop()
    {
        stop.update(ExpStep::Sampling);
        let x = sol.get_sobj().clone_x();
        let id = sol.id();
        state.set_currently(peer, sol);
        let msg = XMsg::<P>(id, x);
        let envelope = Envelope::new(0, peer, msg);
        let res = com.channel(0).send(envelope);
        com.set_busy(peer);
        if let Err(e) = res {
            log::warn!(
                "
                    Failed to send solution ID {:?} to worker {}.
                    The worker will be considered as dead.
                    Error: {:?}
                ",id, peer, &e
            );
           com.set_dead(peer);
           com.set_idle(peer);
           if com.all_dead() {
            return Err(format!("All peers are dead. The last error was {:?}", &e));
           }
        }
    }
    Ok(())
}

fn purge_workers<Com: Communicator>(com: &mut Com) {
    let tolerance = com.size() * 2;
    let mut err_count = 0;
    // while workers are busy, try to flush their channels and mark them as idle
    while com.any_busy() && err_count < tolerance {
        let res = com.channel(0).flush();
        match res {
            Ok(peer) => com.set_idle(peer),
            Err(e) => {
                log::warn!("Error while flushing workers: {}. Continuing...", e);
                err_count += 1;
            },
        }
    }
    if err_count >= tolerance {
        log::warn!("Reached maximum error tolerance while flushing workers. Some workers may still be busy or dead.");
    }
    // Send stop messages to all idle workers
    for peer in com.idle() {
        let msg = Envelope::new(0, peer, StopMsg(0));
        let res = com.channel(0).send(msg);
        if let Err(e) = res {
            log::warn!("Error while sending stop message to worker {}: {}.", peer, e);
        }
    }
    if com.any_busy() {
        log::warn!("
            Unexpected behavior: Some workers are still busy after purging.
            The master will end the experiment, but some workers may not have received the stop message.
            "
        );
    }
}

#[basic_bounds(P)]
fn simple_worker<P, Chan>(
    obj: Arc<P::Func>, 
    chan: &mut Chan, 
) -> Result<(), WorkerError>
where
    P: 'static,
    P::Func: FuncWrapper<Context = (), OutShape = Out<P>>,
    Chan: Channel,
{
    let me = chan.me();
    let msg = Envelope::new(me, 0, InitMsg(0));
    let res = chan.send(msg);
    if let Err(e) = res {
        return Err(WorkerError(format!("Failed to send initialization message from worker {}: {:?}.", me, e)));
    }

    loop {
        let task = receive_with_tolerance(chan, 10);
        if let Ok(msg) = task {
            match msg.tag {
                StopMsg::TAG => break,
                XMsg::<P>::TAG => {
                    let msg = msg.content.into_any().downcast::<XMsg::<P>>().unwrap();
                    let x = msg.as_ref().1.clone();
                    let id = msg.as_ref().0;
                    let outcome = obj.compute(x, ());
                    let outmsg = OutMsg::<P>(id, outcome);
                    let envelope = Envelope::new(me, 0, outmsg);
                    let res = chan.send(envelope);
                    if let Err(e) = res {
                        warn!("Failed to send output message from worker {}: {:?}. Trying to reinitialize the worker.", me, e);
                        let init_msg = Envelope::new(me, 0, InitMsg(0));
                        let res = chan.send(init_msg);
                        if let Err(e) = res {
                            return Err(WorkerError(format!("Failed to reinitialize worker {}: {:?}.", me, e)));
                        }
                    }
                },
                _ => {
                    warn!("Received unexpected message: {:?}. Asking for work...", msg.tag);
                    let envelope = Envelope::new(me, 0, AskMsg(0));
                    let res = chan.send(envelope);
                    if let Err(e) = res {
                        return Err(WorkerError(format!("Failed to send AskMsg from worker {}: {:?}.", me, e)));
                    }
                },
            }
        } else {
            warn!("Failed to receive a message: {:?}", task.err());
            let envelope = Envelope::new(me, 0, AskMsg(0));
            let res = chan.send(envelope);
            if let Err(e) = res {
                return Err(WorkerError(format!("Failed to send AskMsg from worker {}: {:?}.", me, e)));
            }
        }
    }
    Ok(())
}

#[basic_bounds(P)]
#[allow(clippy::too_many_arguments)]
fn simple_master<O, S, R, C, P, Com>(
    mut com: Com,
    problem: P,
    mut accumulator: Accum<P>,
    mut sampler: O,
    mut stop: S,
    mut recorder: Option<R>,
    checkpointer: Option<C>,
    mut state: ParState<P>,
) -> Accum<P>
where
    P: 'static,
    O: Sampler<P, Opt<P>, Info = P::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()> + Send + Sync,
    Com: MasterWorkerCom,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts> + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>] ,&'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
{
    let tolerance = com.size() * 2;
    loop {
        let res = receive_and_update(
            &mut com,
            &problem,
            &mut accumulator,
            &mut sampler,
            &mut stop,
            &mut recorder,
            &mut state,
            tolerance
        );
        if let Err(e) = res {
            log::warn!("Error while receiving messages from workers: {}. Continuing with the experiment.", e);
        } else {
            stop.update(ExpStep::Iteration); // New iteration
        }

        // Check if stop
        if let Some(check) = &checkpointer {
            check.save_state(&state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
            check.save_state(&sampler, &CheckpointerTargets::Optimizer, &()).map_err(|e| log::warn!("{}", e)).ok();
            check.save_state(&stop, &CheckpointerTargets::Stop, &()).map_err(|e| log::warn!("{}", e)).ok();
            check.save_state(&accumulator, &CheckpointerTargets::Accumulator, &()).map_err(|e| log::warn!("{}", e)).ok();
            let parameters = GlobalParameters::new();
            check.save_state(&parameters, &CheckpointerTargets::Config, &()).map_err(|e| log::warn!("{}", e)).ok();
            check.save_state(&state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
        }
        if stop.stop() { break; };

        let res = fill_all_idles(
            &mut com,
            &mut sampler,
            problem.get_searchspace(),
            &mut accumulator,
            &mut stop,
            &mut state
        );
        if let Err(e) = res {
            log::warn!("Error while filling idle workers: {}. Continuing with the experiment.", e);
        }
        
        if let Some(ref check) = checkpointer {
            check.save_state(&state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
        }
    }
    purge_workers(&mut com);
    accumulator
}

/// Start the worker side of a [`SimplePb`] from the current process.
/// See [`MasterWorkerCom::worker`] for the blocking behavior.
fn start_simple_workers<Info, Scp, Out, Com>(
    problem: &SimplePb<Info, Scp, Out>,
    com: &mut Com,
) -> Result<Com::Handle, WorkerError>
where
    SimplePb<Info, Scp, Out>: 'static,
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
    Com: MasterWorkerCom,
{
    let func = problem.clone_objective();
    com.worker(func, move |obj, chan| {
        simple_worker::<SimplePb<Info, Scp, Out>, _>(obj, chan)
    })
}
pub struct Worker<P, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom
{
    /// The [`Problem`] being solved in the experiment.
    pub problem: P,
    /// The [`Checkpointer`] used to save and load experiment state.
    pub checkpointer: Option<C>,
    /// The [`PoolMode`] used for managing function states during the experiment.
    pub pool_mode: PoolMode,
    /// The [`Communicator`] used for send and receiving tasks.
    pub com: Com,
    /// Flag indicating whether to consider Parallel File System (PFS).
    /// Meaning that all processes can access the same file system.
    pub pfs: bool,
    /// Flag indicating whether the experiment state has been loaded from a checkpoint.
    is_loaded: bool,
}

impl<P,C,Com> Worker<P, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom
{
    /// Creates a new `Worker` instance if the current process is not the master.
    /// If it is the master, returns `None`.
    pub fn new(problem: P, checkpointer: Option<C>, pool_mode: PoolMode, com: Com) -> Option<Self> 
    {
        if com.is_master() {
            None
        } else {
            Some(Self {
                problem,
                checkpointer,
                pool_mode,
                com,
                pfs: true,
                is_loaded: false,
            })
        }
    }

    fn load(problem: P, checkpointer: C, pool_mode: PoolMode, com: Com) -> Option<Self>
    {
        if com.is_master(){
            None
        } else {
            Some(Self {
                problem,
                checkpointer: Some(checkpointer),
                pool_mode,
                com,
                pfs: true,
                is_loaded: true,
            })
        }
    }

    /// Set the Parallel File System (PFS) flag for the worker.
    pub fn with_pfs(self, pfs: bool) -> Self {
        Self {
            pfs,
            ..self
        }
    }

    /// Set the [`PoolMode`] for the worker, 
    /// determining how function states are managed during the experiment.
    pub fn with_pool_mode(self, pool_mode: PoolMode) -> Self {
        Self {
            pool_mode,
            ..self
        }
    }
}

impl<Scp, Info, Out, C, Com> WorkerTrait<SimplePb<Info, Scp, Out>> for Worker<SimplePb<Info, Scp, Out>, C, Com>
where
    SimplePb<Info, Scp, Out>: 'static,
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
    C: Checkpointer<SimplePb<Info, Scp, Out>, ()>,
    Com: MasterWorkerCom,
{
    type Handle = Com::Handle;

    fn launch(&mut self) -> Result<Self::Handle, WorkerError> {
        start_simple_workers(&self.problem, &mut self.com)
    }
}

pub struct Master<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom
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
    /// The [`Accumulator`] that collects best evaluated solutions and their outcomes.
    pub accumulator: Accum<P>,
    /// The [`Communicator`] used for send and receiving tasks.
    pub com: Com,
    /// Flag indicating whether to consider Parallel File System (PFS).
    /// Meaning that all processes can access the same file system.
    pub pfs: bool,
    /// Flag indicating whether the experiment state has been loaded from a checkpoint.
    is_loaded: bool,
}

impl<P, O, S, R, C, Com> Master<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom
{
    pub fn new(
        problem: P, 
        sampler: O,
        stop: S, 
        recorder: Option<R>, 
        checkpointer: Option<C>, 
        com: Com
    ) -> Option<Self> {
        if com.is_worker(){
            None
        } else {
            let accumulator = Accum::<P>::default();
            Some(Self {
                problem,
                sampler,
                stop,
                recorder,
                checkpointer,
                accumulator,
                com,
                pfs: true,
                is_loaded: false,
            })
        }
    }

    pub fn load(
        problem: P, 
        mut recorder: Option<R>, 
        mut checkpointer: C, 
        com: Com
    ) -> Result<Self, String> {
        if com.is_worker(){
            Err("Cannot load experiment state on a worker process.".to_string())
        } else {
            if let Some(rec) = recorder.as_mut() {
                rec.before_load(&problem, &()).expect(
                    " 
                        Recorder failed to prepare for loading the experiment state. 
                        Please ensure that the recorder is properly configured and that the necessary files are accessible.
                    "
                );
            }
            checkpointer.before_load(&problem, &()).unwrap();
    
            let sampler: O = checkpointer.load(&CheckpointerTargets::Optimizer, &()).unwrap();
            let stop = checkpointer.load(&CheckpointerTargets::Stop, &()).unwrap();
            let accumulator = checkpointer.load(&CheckpointerTargets::Accumulator, &()).unwrap();
            let parameters: GlobalParameters = checkpointer.load(&CheckpointerTargets::Config, &()).unwrap();
            // if the parameters.id is greater than the current SOL_ID, 
            // we replace SOl_ID by the parameters.id 
            if parameters.sold_id > SOL_ID.load(Ordering::SeqCst) {
                SOL_ID.store(parameters.sold_id, Ordering::SeqCst);
            }
            RUN_ID.store(parameters.run_id + 1, Ordering::SeqCst);

            Ok(Master {
                problem,
                sampler,
                stop,
                recorder,
                checkpointer: Some(checkpointer),
                accumulator,
                com,
                pfs: true,
                is_loaded: true,
            })
        }
    }

    /// Set the Parallel File System (PFS) flag for the master process.
    pub fn with_pfs(self, pfs: bool) -> Self {
        Self {
            pfs,
            ..self
        }
    }
}

impl<O, S, R, C, Scp, Out, Info, Com> Runable for Master<SimplePb<Info, Scp, Out>, O, S, R, C, Com>
where
    SimplePb<Info, Scp, Out>: 'static,
    O:Sampler<SimplePb<Info, Scp, Out>, Scp::Opt, Info = Info>,
    S:Stop,
    R:Recorder<SimplePb<Info, Scp, Out>, ()>,
    C:Checkpointer<SimplePb<Info, Scp, Out>, ()> + Send + Sync,
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
    Com: MasterWorkerCom,
{
    type Id  = SId;
    type Context = ();

    type Problem = SimplePb<Info, Scp, Out>;
    type Sampler = O;
    type Stop = S;
    type Recorder = R;
    type Checkpointer = C;
    
    fn run(mut self) -> Accum<Self::Problem> 
    {   
        let state = if self.is_loaded {
            if let Some(rec) = &mut self.recorder{
                rec.after_load(&self.problem, &()).unwrap();
            }

            if let Some(check) = &mut self.checkpointer{
                check.after_load(&self.problem, &()).unwrap();
                let state = check.load(&CheckpointerTargets::Experiment, &());    
                match state {
                    Ok(e) => e,
                    Err(e) => {
                        log::warn!("Failed to load experiment state: {}.", e);
                        ParState::new(self.com.size() + 1)
                    },
                }
            } else {
                ParState::new(self.com.size() + 1)
            }
        } else {
            if let Some(rec) = &mut self.recorder{
                rec.init(&self.problem, &()).unwrap();
            }
            if let Some(check) = &mut self.checkpointer{
                check.init(&self.problem, &()).unwrap();
            }
            ParState::new(self.com.size() + 1)
        };


        let problem = self.problem;
        let accumulator = self.accumulator;
        let sampler = self.sampler;
        let stop = self.stop;
        let recorder = self.recorder;
        let checkpointer = self.checkpointer;

        self.com.master::<SimplePb<Info, Scp, Out>,_>(
            move |com|
            simple_master(
                com, 
                problem, 
                accumulator, 
                sampler, 
                stop, 
                recorder, 
                checkpointer, 
                state
            )
        )
    }
    
    fn get_stop(&self) -> &Self::Stop {
        &self.stop
    }
    
    fn searchspace(&self) -> &crate::aliases::Scp<Self::Problem> {
        self.problem.get_searchspace()
    }
    
    fn codomain(&self) -> &Cod<Self::Problem> {
        self.problem.get_codomain()
    }
    
    fn objective(&self) -> &crate::aliases::Func<Self::Problem> {
        self.problem.get_objective()
    }
    
    fn sampler(&self) -> &Self::Sampler {
        &self.sampler
    }
    
    fn recorder(&self) -> Option<&Self::Recorder> {
        self.recorder.as_ref()
    }
    
    fn checkpointer(&self) -> Option<&Self::Checkpointer> {
        self.checkpointer.as_ref()
    }

    fn accumulator(&self) -> &Accum<Self::Problem> {
        &self.accumulator
    }
    
    fn mut_stop(&mut self) -> &mut Self::Stop {
        &mut self.stop
    }
    
    fn mut_searchspace(&mut self) -> &mut crate::aliases::Scp<Self::Problem> {
        self.problem.get_mut_searchspace()
    }
    
    fn mut_codomain(&mut self) -> &mut Cod<Self::Problem> {
        self.problem.get_mut_codomain()
    }
    
    fn mut_objective(&mut self) -> &mut crate::aliases::Func<Self::Problem> {
        self.problem.get_mut_objective()
    }
    
    fn mut_sampler(&mut self) -> &mut Self::Sampler {
        &mut self.sampler
    }
    
    fn mut_recorder(&mut self) -> Option<&mut Self::Recorder> {
        self.recorder.as_mut()
    }
    
    fn mut_checkpointer(&mut self) -> Option<&mut Self::Checkpointer> {
        self.checkpointer.as_mut()
    }
    
    fn mut_accumulator(&mut self) -> &mut Accum<Self::Problem> {
        &mut self.accumulator
    }
}

enum Role<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    Master(Master<P, O, S, R, C, Com>),
    Worker(Worker<P, C, Com>),
}

/// A parallel experiment, with the same user experience as [`Experiment`].
///
/// It hides the [`Worker`] / [`Master`] split and guarantees that the workers are started
/// before the master routine runs, whatever the [`MasterWorkerCom`] backend:
/// - threaded: the worker threads are spawned, the master routine runs in the calling thread,
///   then the threads are joined.
/// - MPI: the same code is executed by every rank. Worker ranks run the worker routine
///   and never return, the master rank runs the master routine.
///
/// ```ignore
/// let exp = ParExperiment::new(pb, opt, stop, rec, check, ThreadCommunicator::new());
/// let accumulator = exp.run(); // `Some(_)` on the master, `None` on MPI worker ranks.
///
/// let exp = ParExperiment::<_, RandomSearch, Evaluated, _, _, _>::load(pb, rec, check, com);
/// ```
pub struct ParExperiment<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    role: Role<P, O, S, R, C, Com>,
}

impl<P, O, S, R, C, Com> ParExperiment<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    /// Create a new parallel experiment.
    pub fn new(
        problem: P,
        sampler: O,
        stop: S,
        recorder: Option<R>,
        checkpointer: Option<C>,
        com: Com,
    ) -> Self {
        let role = if com.is_master() {
            Role::Master(
                Master::new(problem, sampler, stop, recorder, checkpointer, com)
                    .expect("A master process must be able to create a Master."),
            )
        } else {
            Role::Worker(
                Worker::new(problem, checkpointer, PoolMode::InMemory, com)
                    .expect("A worker process must be able to create a Worker."),
            )
        };
        Self { role }
    }

    /// Load a parallel experiment from a [`Checkpointer`].
    /// Only the master process reads the checkpoint.
    pub fn load(problem: P, recorder: Option<R>, checkpointer: C, com: Com) -> Self {
        let role = if com.is_master() {
            Role::Master(Master::load(problem, recorder, checkpointer, com).unwrap())
        } else {
            Role::Worker(
                Worker::load(problem, checkpointer, PoolMode::InMemory, com)
                    .expect("A worker process must be able to create a Worker."),
            )
        }; 
        Self { role }
    }

    /// Set the Parallel File System (PFS) flag.
    pub fn with_pfs(mut self, pfs: bool) -> Self {
        match &mut self.role {
            Role::Master(m) => m.pfs = pfs,
            Role::Worker(w) => w.pfs = pfs,
        }
        self
    }

    /// Return the [`Master`] if the current process is the master, to inspect or modify the experiment before running.
    pub fn master_mut(&mut self) -> Option<&mut Master<P, O, S, R, C, Com>> {
        match &mut self.role {
            Role::Master(m) => Some(m),
            Role::Worker(_) => None,
        }
    }
}

impl<O, S, R, C, Scp, Out, Info, Com> ParExperiment<SimplePb<Info, Scp, Out>, O, S, R, C, Com>
where
    SimplePb<Info, Scp, Out>: 'static,
    O: Sampler<SimplePb<Info, Scp, Out>, Scp::Opt, Info = Info>,
    S: Stop,
    R: Recorder<SimplePb<Info, Scp, Out>, ()>,
    C: Checkpointer<SimplePb<Info, Scp, Out>, ()> + Send + Sync,
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
    Com: MasterWorkerCom,
{
    /// Run the experiment. Workers are always started before the master routine.
    ///
    /// Returns the [`Accumulator`] on the master, and `None` on worker processes.
    pub fn run(self) -> Option<Accum<SimplePb<Info, Scp, Out>>> {
        match self.role {
            Role::Master(mut master) => {
                let handle = start_simple_workers(&master.problem, &mut master.com)
                    .map_err(|e| log::error!("Failed to start workers: {}", e))
                    .ok()?;
                let accumulator = master.run();
                if let Err(e) = handle.join() {
                    log::error!("A worker failed: {}", e);
                }
                Some(accumulator)
            }
            Role::Worker(mut worker) => {
                if let Err(e) = worker.launch().and_then(|h| h.join()) {
                    log::error!("A worker failed: {}", e);
                }
                None
            }
        }
    }
}
