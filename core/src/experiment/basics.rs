use std::{marker::PhantomData, process::exit, sync::atomic::Ordering};

use crate::{
    Accumulator, BaseSol, Checkpointer, Codomain, Computed, FidOutcome, FidPb, FidelitySol, FuncState, FuncWrapper, GlobalParameters, HasFidelity, HasId, HasState, HasStep, HasStepId, HasVariables, HasX, MultiStep, OneStep, Outcome, Problem, RUN_ID, Runable, SId, SOL_ID, Sampler, Searchspace, SimplePb, SolId, SolInfo, SolutionShape, State, Step, StepId, StepSId, Stop, Var, aliases::{Accum, Cod, CompShape, LinkObj, LinkOpt, Obj, Opt, Out, SObj, SOpt, Shape, TrueOpt}, checkpointer::CheckpointerTargets, experiment::PoolMode, objective::outcome::IdOut, recorder::{Recorder, RecorderTargets}, solution::shape::IntoComputedShape, stop::ExpStep, utils::record::{CompParts, Recordable, Schema, SolParts}
};

use indexmap::IndexMap;
use log::warn;
use serde::{Deserialize, Serialize};

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
    pub problem: P,
    pub sampler: O,
    pub stop: S,
    pub recorder: Option<R>,
    pub checkpointer: Option<C>,
    pub accumulator: Accum<P>,
    pub pool_mode: PoolMode,
    pub state: ExpState<P>,
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
        SOL_ID.store(parameters.sold_id, Ordering::Relaxed);
        RUN_ID.store(parameters.run_id + 1, Ordering::Relaxed);

        if let Some(rec) = recorder.as_mut() {
            rec.after_load(&problem, &()).unwrap();
        }
        checkpointer.after_load(&problem, &()).unwrap();

        let exp = checkpointer.load(&CheckpointerTargets::Experiment, &());

        let state = match exp {
            Ok(e) => e,
            Err(e) => {
                log::warn!("
                    Failed to load experiment state: {}. Because of an error, or because
                    the experiment unintentionally stopped after an evaluation and before sampling a new solution.
                    The experiment will now attempt to recover by sampling a new solution.
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
        rec.record(&RecorderTargets::Outcome, &IdOut(&computed.id(),outputed), (&(),&()), ()).unwrap();
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

// /// An enum that abstracts over different implementations of a function state pool.
// /// It can either be an in-memory [`IdxMapPool`] or a checkpointer-backed [`LoadPool`].
// /// This allows the experiment to switch between different pool strategies based on the [`PoolMode`] without changing the underlying logic of function state management.
// pub enum Pool<FnStCheck, FnState, SolId>
// where
//     FnState: FuncState,
//     SolId: Id,
//     FnStCheck: FuncStateCheckpointer,
// {
//     IdxMap(IdxMapPool<FnStCheck, FnState, SolId>),
//     Load(LoadPool<FnStCheck, FnState, SolId>),
// }

// impl<FnStCheck, FnState, SolId> Default for Pool<FnStCheck, FnState, SolId>
// where
//     FnState: FuncState,
//     SolId: Id,
//     FnStCheck: FuncStateCheckpointer,
// {
//     fn default() -> Self {
//         Pool::IdxMap(IdxMapPool::default())
//     }
// }

// impl<FnStCheck, FnState, SolId> FuncStatePool<FnState, SolId> for Pool<FnStCheck, FnState, SolId>
// where
//     FnState: FuncState,
//     SolId: Id,
//     FnStCheck: FuncStateCheckpointer,
// {
//     /// Insert a new [`FuncState`] into the pool, associated with the given solution [`StepId`].
//     /// The behavior of this method depends on the underlying pool implementation:
//     /// - For [`IdxMapPool`], it will save the state in the in-memory map and optionally persist it using the checkpointer.
//     /// - For [`LoadPool`], it will save the state using the checkpointer and keep track of the current state in memory for quick retrieval
//     fn insert(&mut self, id: SolId, state: FnState) {
//         match self {
//             Pool::IdxMap(p) => p.insert(id, state),
//             Pool::Load(p) => p.insert(id, state),
//         }
//     }

//     /// Remove the [`FuncState`] associated with the given solution [`StepId`], if it exists.
//     /// The behavior of this method depends on the underlying pool implementation:
//     /// - For [`IdxMapPool`], it will remove the state from the in-memory map and optionally remove it from the checkpointer.
//     /// - For [`LoadPool`], it will remove the state from the checkpointer and clear the current state if it matches the given [`StepId`].
//     fn remove(&mut self, id: &SolId) -> bool {
//         match self {
//             Pool::IdxMap(p) => p.remove(id),
//             Pool::Load(p) => p.remove(id),
//         }
//     }

//     /// Retrieve the [`FuncState`] associated with the given solution [`StepId`], if it exists.
//     /// The behavior of this method depends on the underlying pool implementation:
//     /// - For [`IdxMapPool`], it will retrieve the state from the in-memory map.
//     /// - For [`LoadPool`], it will first check if the current state in memory matches the given [`StepId`] and return it if so;
//     ///   otherwise, it will attempt to load the state from the checkpointer.
//     fn retrieve(&mut self, id: &SolId) -> Option<FnState> {
//         match self {
//             Pool::IdxMap(p) => p.retrieve(id),
//             Pool::Load(p) => p.retrieve(id),
//         }
//     }
// }



// /// Creates a single-threaded experiment ([`MonoExperiment`]).
// ///
// /// This is the primary entry point for creating experiments that execute in a single thread.
// /// The experiment combines all necessary components and can be immediately run or inspected.
// ///
// /// # Parameters
// ///
// /// * `space` - A [`Searchspace`], $f: X \rightarrow Y$
// /// * `objective` - The objective function wrapped in [`Objective`](crate::objective::Objective) or [`Stepped`](crate::objective::Stepped), $f$
// /// * `optimizer` - An optimizer implementing [`Optimizer`]
// /// * `stop` - A stopping criterion implementing [`Stop`]
// /// * `saver` - A tuple of (optional [`Recorder`], optional [`Checkpointer`])
// ///
// /// # Returns
// ///
// /// An experiment implementing [`Runable`] ready to be executed
// ///
// /// # Example
// ///
// /// ```rust,ignore
// /// use tantale::core::{experiment::mono, Objective, stop::Calls};
// /// use tantale::algos::RandomSearch;
// ///
// /// let sp = my_searchspace();
// /// let obj = Objective::new(my_function);
// /// let opt = RandomSearch::new();
// /// let stop = Calls::new(100);
// ///
// /// let experiment = mono(sp, obj, opt, stop, (None, None)); // No recording or checkpointing
// /// experiment.run();
// /// ```
// ///
// /// # See Also
// ///
// /// * [`mono_with_pool`] - For single-threaded execution with function state management.
// /// * [`threaded`] - For multi-threaded execution
// /// * [`distributed`] - For MPI-based distributed execution
// /// * [`MonoExperiment`] - The underlying experiment type
// pub fn mono<Scp, Func, Opt, St, Rec, Check>(
//     space: Scp,
//     objective: Func,
//     optimizer: Opt,
//     stop: St,
//     rec : Option<Rec>,
//     check: Option<Check>,
// ) -> impl Runable<BasicProblem<Scp, Func>>
// where
//     Scp: Searchspace<Func>,
//     Func: FuncWrapper,
//     Opt: Optimizer<BasicProblem<Scp, Func>>,
//     St: Stop,
//     Rec: Recorder,
//     Check: Checkpointer,
// {
//     <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new(
//         space, objective, optimizer, stop, saver,
//     )
// }

// /// Similar to [`mono`] but allows specifying a [`PoolMode`] for function state management.
// pub fn mono_with_pool<SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     optimizer: Op,
//     stop: St,
//     saver: (Option<Rec>, Option<Check>),
//     pool_mode: PoolMode,
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     St: Stop,
//     Rec: Recorder,
//     Check: MonoCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new_with_pool(
//         space, objective, optimizer, stop, saver, pool_mode,
//     )
// }

// /// Creates a multi-threaded experiment ([`ThrExperiment`]).
// ///
// /// This function creates experiments that execute using multiple OS threads, enabling
// /// parallel evaluation of solutions. Particularly effective for batch optimizers or
// /// sequential optimizers with varying evaluation times.
// ///
// /// # Parameters
// ///
// /// * `space` - A [`Searchspace`]
// /// * `objective` - The objective function wrapped in [`Objective`](crate::objective::Objective) or [`Stepped`](crate::objective::Stepped)
// /// * `optimizer` - An optimizer implementing [`Optimizer`]
// /// * `stop` - A stopping criterion implementing [`Stop`]
// /// * `saver` - A tuple of (optional [`Recorder`], optional [`ThrCheckpointer`])
// ///
// /// # Thread Safety
// ///
// /// All components must implement `Send + Sync` for thread-safe sharing across threads.
// ///
// /// # Example
// ///
// /// ```rust,ignore
// /// use tantale::core::{experiment::threaded, Objective, stop::Calls};
// /// use tantale::algos::BatchRandomSearch;
// ///
// /// let sp = my_searchspace();
// /// let obj = Objective::new(my_function);
// /// let opt = BatchRandomSearch::new(10);  // Batch size 10
// /// let stop = Calls::new(100);
// ///
// /// let experiment = threaded(sp, obj, opt, stop, (None, None)); // No recording or checkpointing
// /// experiment.run();  // Evaluates batches in parallel
// /// ```
// ///
// /// # See Also
// ///
// /// * [`threaded_with_pool`] - For multi-threaded execution with function state management.
// /// * [`mono`] - For single-threaded execution
// /// * [`distributed`] - For MPI-based distributed execution
// /// * [`ThrExperiment`] - The underlying experiment type
// pub fn threaded<SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     optimizer: Op,
//     stop: St,
//     saver: (Option<Rec>, Option<Check>),
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     St: Stop,
//     Rec: Recorder,
//     Check: ThrCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new(
//         space, objective, optimizer, stop, saver,
//     )
// }

// /// Similar to [`threaded`] but allows specifying a [`PoolMode`] for function state management.
// pub fn threaded_with_pool<SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     optimizer: Op,
//     stop: St,
//     saver: (Option<Rec>, Option<Check>),
//     pool_mode: PoolMode,
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     St: Stop,
//     Rec: Recorder,
//     Check: ThrCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new_with_pool(
//         space, objective, optimizer, stop, saver, pool_mode,
//     )
// }

// #[cfg(feature = "mpi")]
// #[allow(clippy::type_complexity)]
// /// Creates an MPI-distributed experiment ([`MPIExperiment`]).
// ///
// /// This function creates experiments that execute across multiple processes using MPI, enabling
// /// parallel evaluation of solutions in a distributed environment.
// /// Requires the `mpi` feature and an initialized MPI environment.
// /// Each process will run an instance of the experiment, with one process acting as the master coordinating the optimization
// /// and others as workers evaluating solutions.
// ///
// /// # Parameters
// ///
// /// * `proc` - A reference to an initialized [`MPIProcess`] containing rank, communicator, etc.
// /// * `space` - A [`Searchspace`]
// /// * `objective` - The objective function wrapped in [`Objective`](crate::objective::Objective) or [`Stepped`](crate::objective::Stepped)
// /// * `optimizer` - An optimizer implementing [`Optimizer`]
// /// * `stop` - A stopping criterion implementing [`Stop`]
// /// * `saver` - A tuple of (optional [`Recorder`], optional [`ThrCheckpointer`])
// ///
// ///
// /// # Example
// ///
// /// ```rust,ignore
// /// use tantale::core::{experiment::threaded, Objective, stop::Calls};
// /// use tantale::algos::BatchRandomSearch;
// ///
// /// let sp = my_searchspace();
// /// let obj = Objective::new(my_function);
// /// let opt = BatchRandomSearch::new(10);  // Batch size 10
// /// let stop = Calls::new(100);
// ///
// /// let experiment = distributed(&proc, sp, obj, opt, stop, (None, None)); // No recording or checkpointing
// /// experiment.run();  // Evaluates batches in parallel
// /// ```
// ///
// /// # See Also
// ///
// /// * [`distributed_with_pool`] - For MPI-based distributed execution with function state management.
// /// * [`mono`] - For single-threaded execution
// /// * [`threaded`] - For multi-threaded execution
// /// * [`MPIExperiment`] - The underlying experiment type
// pub fn distributed<'a, SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
//     proc: &'a MPIProcess,
//     space: Scp,
//     objective: Fn,
//     optimizer: Op,
//     stop: St,
//     saver: (Option<Rec>, Option<Check>),
// ) -> MasterWorker<
//     'a,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
//     PSol,
//     SolId,
//     Scp,
//     Op,
//     St,
//     Rec,
//     Check,
//     Out,
//     Fn,
// >
// where
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     St: Stop,
//     Rec: Recorder,
//     Check: DistCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::new(
//         proc,
//         space, objective, optimizer, stop, saver
//     )
// }

// #[cfg(feature = "mpi")]
// #[allow(clippy::type_complexity)]
// /// Similar to [`distributed`] but allows specifying a [`PoolMode`] for function state management.
// pub fn distributed_with_pool<'a, SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
//     proc: &'a MPIProcess,
//     space: Scp,
//     objective: Fn,
//     optimizer: Op,
//     stop: St,
//     saver: (Option<Rec>, Option<Check>),
//     pool_mode: PoolMode,
// ) -> MasterWorker<
//     'a,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
//     PSol,
//     SolId,
//     Scp,
//     Op,
//     St,
//     Rec,
//     Check,
//     Out,
//     Fn,
// >
// where
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     St: Stop,
//     Rec: Recorder,
//     Check: DistCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::new_with_pool(
//         proc,
//         space, objective, optimizer, stop, saver, pool_mode
//     )
// }

// /// Loads a single-threaded experiment from a checkpoint.
// ///
// /// Restores a [`MonoExperiment`] from saved state, allowing optimization to be resumed
// /// or inspected. All optimizer state, stopping criterion progress, and relevant data
// /// are recovered from the checkpoint.
// ///
// /// # Parameters
// ///
// /// * `space` - A [`Searchspace`] (must be the same as used for the original experiment)
// /// * `objective` - The objective function wrapper  
// /// * `saver` - Tuple of (optional [`Recorder`], **required** [`Checkpointer`])
// ///
// /// # Type Parameters
// ///
// /// Unlike [`mono`], you must explicitly specify the optimizer and stop types:
// /// - `Op` - The optimizer type
// /// - `St` - The stopping criterion type
// ///
// /// # Returns
// ///
// /// A loaded experiment implementing [`Runable`], ready to continue or be inspected
// ///
// /// # Example
// ///
// /// ```rust,ignore
// /// use tantale::core::{experiment::mono_load, MessagePack, FolderConfig};
// /// use tantale::algos::RandomSearch;
// /// use tantale::core::stop::Calls;
// ///
// /// // Load from checkpoint
// /// let config = FolderConfig::new("my_experiment").init();
// /// let checkpointer = MessagePack::new(config).unwrap();
// ///
// /// let mut exp = mono_load::<RandomSearch, Calls, _, _, _, _, _, _, _>(
// ///     searchspace,
// ///     objective,
// ///     (None, checkpointer)
// /// );
// ///
// /// // Inspect or modify before continuing
// /// println!("Completed {} evaluations", exp.get_stop().calls());
// /// exp.get_mut_stop().add(100);  // Add 100 more evaluations
// ///
// /// // Continue optimization
// /// exp.run();
// /// ```
// ///
// /// # Note
// ///
// /// Prefer using the `load!` macro for cleaner syntax and better type inference.
// ///
// /// # See Also
// ///
// /// * [`mono`] - For creating new experiments
// /// * [`threaded_load`] - For loading multi-threaded experiments
// /// * `load!` - Macro for simpler loading syntax
// pub fn mono_load<SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     saver: (Option<Rec>, Check),
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     SolId: Id,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     St: Stop,
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     Rec: Recorder,
//     Check: MonoCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
// {
//     <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load(
//         space, objective, saver,
//     )
// }

// /// Similar to [`mono_load`] but allows specifying a [`PoolMode`] for function state management.
// pub fn mono_load_with_pool<SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     saver: (Option<Rec>, Check),
//     pool_mode: PoolMode,
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     SolId: Id,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     St: Stop,
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     Rec: Recorder,
//     Check: MonoCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
// {
//     <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load_with_pool(
//         space, objective, saver, pool_mode
//     )
// }

// /// Loads a multi-threaded experiment from a checkpoint.
// ///
// /// Restores a [`ThrExperiment`] from saved state. The checkpoint must have been created
// /// with a thread-safe checkpointer ([`ThrCheckpointer`]).
// ///
// /// # Parameters
// ///
// /// * `space` - A [`Searchspace`] (must be the same as used for the original experiment)
// /// * `objective` - The objective function wrapper
// /// * `saver` - Tuple of (optional [`Recorder`], **required** [`ThrCheckpointer`])
// ///
// /// # Type Parameters
// ///
// /// - `Op` - The optimizer type
// /// - `St` - The stopping criterion type
// ///
// /// # Example
// ///
// /// ```rust,ignore
// /// use tantale::core::{experiment::threaded_load, MessagePack, FolderConfig};
// /// use tantale::algos::BatchRandomSearch;
// /// use tantale::core::stop::Calls;
// ///
// /// let config = FolderConfig::new("my_experiment").init();
// /// let checkpointer = MessagePack::new(config).unwrap();
// ///
// /// let mut exp = threaded_load::<BatchRandomSearch, Calls, _, _, _, _, _, _, _>(
// ///     searchspace,
// ///     objective,
// ///     (None, checkpointer)
// /// );
// ///
// /// exp.run();
// /// ```
// ///
// /// # Note
// ///
// /// Prefer using the `load!` macro for cleaner syntax.
// ///
// /// # See Also
// ///
// /// * [`threaded`] - For creating new multi-threaded experiments
// /// * [`mono_load`] - For loading single-threaded experiments
// /// * `load!` - Macro for simpler loading syntax
// pub fn threaded_load<SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     saver: (Option<Rec>, Check),
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     SolId: Id,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     St: Stop,
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     Rec: Recorder,
//     Check: ThrCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
// {
//     <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load(
//         space, objective, saver,
//     )
// }

// /// Similar to [`threaded_load`] but allows specifying a [`PoolMode`] for function state management.
// pub fn threaded_load_with_pool<SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     saver: (Option<Rec>, Check),
//     pool_mode: PoolMode,
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     SolId: Id,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     St: Stop,
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     Rec: Recorder,
//     Check: ThrCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
// {
//     <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load_with_pool(
//         space, objective, saver, pool_mode
//     )
// }

// #[cfg(feature = "mpi")]
// #[allow(clippy::type_complexity)]
// /// Loads an MPI-distributed experiment from a checkpoint.
// ///
// /// Restores a [`MPIExperiment`] from saved state. The checkpoint must have been created
// /// with a distributed checkpointer ([`DistCheckpointer`]) and requires the `mpi`
// /// feature and an initialized MPI environment. Each process will load its own instance of the experiment,
// /// with one process acting as the master coordinating the optimization and others as workers evaluating solutions.
// ///
// /// # Parameters
// ///
// /// * `proc` - A reference to an initialized [`MPIProcess`] containing rank, communicator, etc.
// /// * `space` - A [`Searchspace`] (must be the same as used for the original experiment)
// /// * `objective` - The objective function wrapper
// /// * `saver` - Tuple of (optional [`Recorder`], **required** [`DistCheckpointer`])
// ///
// /// # Type Parameters
// ///
// /// - `Op` - The optimizer type
// /// - `St` - The stopping criterion type
// ///
// /// # Example
// ///
// /// ```rust,ignore
// /// use tantale::core::{experiment::distributed_load, MessagePack, FolderConfig};
// /// use tantale::core::mpi::utils::MPIProcess;
// /// use tantale::algos::BatchRandomSearch;
// /// use tantale::core::stop::Calls;
// ///
// /// let config = FolderConfig::new("my_experiment").init();
// /// let checkpointer = MessagePack::new(config).unwrap();
// /// let proc = MPIProcess::new();
// ///
// /// let mut exp = distributed_load::<BatchRandomSearch, Calls, _, _, _, _, _, _, _>(
// ///     &proc,
// ///     searchspace,
// ///     objective,
// ///     (None, checkpointer)
// /// );
// ///
// /// exp.run();
// /// ```
// ///
// /// # Note
// ///
// /// Prefer using the `load!` macro for cleaner syntax.
// ///
// /// # See Also
// ///
// /// * [`distributed`] - For creating new multi-threaded experiments
// /// * [`mono_load`] - For loading single-threaded experiments
// /// * `load!` - Macro for simpler loading syntax
// pub fn distributed_load<'a, SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
//     proc: &'a MPIProcess,
//     space: Scp,
//     objective: Fn,
//     saver: (Option<Rec>, Check),
// ) -> MasterWorker<
//     'a,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
//     PSol,
//     SolId,
//     Scp,
//     Op,
//     St,
//     Rec,
//     Check,
//     Out,
//     Fn,
// >
// where
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     St: Stop,
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     Rec: Recorder,
//     Check: DistCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::load(
//         proc,
//         space,
//         objective,
//         saver
//     )
// }

// #[cfg(feature = "mpi")]
// #[allow(clippy::type_complexity)]
// /// Similar to [`distributed_load`] but allows specifying a [`PoolMode`] for function state management.
// pub fn distributed_load_with_pool<'a, SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
//     proc: &'a MPIProcess,
//     space: Scp,
//     objective: Fn,
//     saver: (Option<Rec>, Check),
//     pool_mode: PoolMode,
// ) -> MasterWorker<
//     'a,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
//     PSol,
//     SolId,
//     Scp,
//     Op,
//     St,
//     Rec,
//     Check,
//     Out,
//     Fn,
// >
// where
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     St: Stop,
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     Rec: Recorder,
//     Check: DistCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::load_with_pool(
//         proc,
//         space,
//         objective,
//         saver,
//         pool_mode
//     )
// }