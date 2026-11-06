use std::marker::PhantomData;

use crate::{
    Checkpointer, Computed, FuncState, FuncWrapper, 
    HasId, HasVariables, Problem, 
    SolId, SolInfo, SolutionShape, StepId, Var, aliases::{Cod, CompShape, LinkObj, Obj, Out, 
    SObj, SOpt, TrueOpt},
    objective::outcome::IdOut, recorder::{Recorder, RecorderTargets},
    utils::record::{CompParts, Recordable, Schema, SolParts},
};

use indexmap::IndexMap;

/// Record the computed solution and the output of the problem using the provided recorder.
pub fn record<R: Recorder<P,Ctx>, P: Problem, Ctx>
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

/// Pool modes for the [`FuncStatePool`](crate::experiment::basics::FuncStatePool) when retrieving [`FuncState`](crate::FuncState).
/// This prevent overflowing RAM whith large [`FuncState`](crate::FuncState) by allowing to retrieve the state from memory if it is still there,
/// or to load it from disk if it has been evicted.
/// It is only used for multi-fidelity optimization where the state of the objective function has to be tracked.
#[derive(Debug, Clone, Copy)]
pub enum PoolMode {
    /// Retrieve the state saved in memory (if any) without loading from disk.
    InMemory,
    /// Load the state from disk using the checkpointer.
    Persistent,
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
    /// Remove all previous [`FuncState`]s associated with the given solution [`StepId`], if they exist.
    fn remove_steps<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) {
        let mut prev_id = *id;
        self.remove::<P, Ctx, C>(&prev_id, check, ctx);
        while prev_id.id_step() > 0 {
            prev_id = prev_id.previous_id();
            self.remove::<P, Ctx, C>(&prev_id, check, ctx);
        }
    }
    /// Tells which IDs are stored in the pool. This is useful for debugging and monitoring the state of the pool.
    fn stored_ids<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, check: &Option<C>,ctx: &Ctx) -> Vec<Id>;
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
            let _ = c.remove_fn_state::<Id, FnState>(id, ctx);
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
    /// Returns a vector of all solution [`StepId`]s currently stored in the pool.
    fn stored_ids<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, _check: &Option<C>, _ctx: &Ctx) -> Vec<Id> {
        self.0.keys().copied().collect()
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

    /// Inspect the persistent storage to determine which solution [`StepId`]s have associated [`FuncState`]s stored.
    /// This implementation relies on the checkpointer to provide the list of stored IDs.
    fn stored_ids<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, check: &Option<C>,ctx: &Ctx) -> Vec<Id> {
        if let Some(c) = check {
            c.list_ids(ctx).unwrap_or_default()
        } else {
            panic!("LoadPool requires a checkpointer to list stored function states.");
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

    fn stored_ids<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, check: &Option<C>,ctx: &Ctx) -> Vec<Id> {
        match self {
            Pool::InMemory(p) => p.stored_ids(check, ctx),
            Pool::Persistent(p) => p.stored_ids(check, ctx),
        }
    }
}