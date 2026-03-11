use crate::{
    FidOutcome, HasFidelity, HasStep,
    domain::{Codomain, Domain, onto::LinkOpt},
    experiment::CompAcc,
    objective::{FuncWrapper, Outcome},
    recorder::csv::CSVWritable,
    searchspace::{CompShape, OptionCompShape, Searchspace},
    solution::{Batch, Id, IntoComputed, SolInfo, SolutionShape, Uncomputed, shape::RawObj},
};

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Per-iteration metadata produced by an [`Optimizer`].
///
/// This typically aggregates informations about
/// the current iteration of the [`Optimizer`].
///
/// # Associated Derive Macro
///
/// The `OptInfo` derive macro automatically implements the trait for any struct
/// satisfying the required trait bounds.
pub trait OptInfo
where
    Self: Serialize + for<'de> Deserialize<'de> + CSVWritable<(), ()> + Debug + Default,
{
}

/// Serializable state of an [`Optimizer`].
///
/// Implementations should capture all information required to resume an
/// optimization after checkpointing.
///
/// # Associated Derive Macro
///
/// The `OptState` derive macro automatically implements the trait for any struct
/// satisfying the required trait bounds.
pub trait OptState
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}

/// Empty implementation for [`OptInfo`] or [`SolInfo`].
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct EmptyInfo;
impl SolInfo for EmptyInfo {}
impl OptInfo for EmptyInfo {}
impl CSVWritable<(), ()> for EmptyInfo {
    fn header(_elem: &()) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::new()
    }
}

/// Type aliases for cleaner associated type definitions of [`Optimizer`]s [`SolInfo`].
pub type OpSInfType<Op, PSol, Scp, SolId, Out> =
    <Op as Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>>::SInfo;
/// Type aliases for cleaner associated type definitions of [`Optimizer`]s [`Codomain`].
pub type OpCodType<Op, PSol, Scp, SolId, Out> =
    <Op as Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>>::Cod;

/// Core optimizer abstraction.
///
/// An [`Optimizer`] generates candidate [`Solution`](crate::solution::Solution)s to
/// **maximize** an objective, and maintains the state required to resume or
/// checkpoint an experiment.
pub trait Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
{
    type State: OptState;
    type Cod: Codomain<Out>;
    type SInfo: SolInfo;

    /// Returns a mutable reference to the [`OptState`] of the [`Optimizer`].
    fn get_mut_state(&mut self) -> &Self::State;
    /// Returns a reference to the [`OptState`] of the [`Optimizer`].
    fn get_state(&self) -> &Self::State;
    /// Builds an [`Optimizer`] from a previously saved [`OptState`].
    fn from_state(state: Self::State) -> Self;
}

/// A [`Batch`] of [`CompShape`](crate::searchspace::CompShape) solutions for a given [`Searchspace`] and [`Codomain`].
pub type CompBatch<SolId, SInfo, Info, Scp, PSol, Cod, Out> =
    Batch<SolId, SInfo, Info, CompShape<Scp, PSol, SolId, SInfo, Cod, Out>>;

/// Batch optimizer interface.
///
/// At each iteration, the optimizer produces a whole [`Batch`] of new candidates.
pub trait BatchOptimizer<PSol, SolId, Opt, Out, Scp, Fn>:
    Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>: SolutionShape<SolId, Self::SInfo>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Self::SInfo>>,
{
    type Info: OptInfo + Send + Sync;

    /// Executed once at the beginning of the optimization. Does not require previous
    /// computed solutions.
    fn first_step(&mut self, scp: &Scp) -> Batch<SolId, Self::SInfo, Self::Info, Scp::SolShape>;
    /// Computes a single iteration of the [`Optimizer`].
    ///
    /// It consumes a [`Batch`] of computed solutions and returns a new batch of
    /// uncomputed candidates.
    ///
    /// The `acc` parameter provides a view of the best solutions
    /// accumulated since the start of the experiment:
    /// - For single-objective [`Codomain`](crate::Codomain)s, `acc` is a
    ///   [`BestComputed`](crate::domain::codomain::BestComputed) holding the single best solution seen so far.
    /// - For multi-objective [`Codomain`](crate::Codomain)s, `acc` is a
    ///   [`ParetoComputed`](crate::domain::codomain::ParetoComputed) holding the current Pareto front.
    ///
    /// The accumulator is maintained externally by the [`Runable`](crate::Runable) and
    /// updated after each batch evaluation. The [`Optimizer`] should use it read-only
    /// to inform the generation of new candidates (e.g. for surrogate models or
    /// elitist strategies).
    fn step(
        &mut self,
        x: CompBatch<SolId, Self::SInfo, Self::Info, Scp, PSol, Self::Cod, Out>,
        scp: &Scp,
        acc: &CompAcc<Scp, PSol, SolId, Self::SInfo, Self::Cod, Out>,
    ) -> Batch<SolId, Self::SInfo, Self::Info, Scp::SolShape>;

    /// Sets the batch size for the optimizer.
    fn set_batch_size(&mut self, batch_size: usize);

    /// Returns the current batch size of the optimizer.
    fn get_batch_size(&self) -> usize;
}

/// Sequential optimizer interface.
///
/// At each iteration, the optimizer produces a single [`Uncomputed`] candidate.
pub trait SequentialOptimizer<PSol, SolId, Opt, Out, Scp, Fn>:
    Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>: SolutionShape<SolId, Self::SInfo>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Self::SInfo>>,
{
    /// Computes a single iteration of the [`Optimizer`].
    ///
    /// It consumes an optional computed solution and returns a new uncomputed
    /// candidate. [`Self`] is mutable to update the internal state.
    ///
    /// The `acc` parameter provides a read-only view of the best solutions
    /// accumulated since the start of the experiment:
    /// - For single-objective [`Codomain`](crate::Codomain)s, `acc` is a
    ///   [`BestComputed`](crate::BestComputed) holding the single best solution seen so far.
    /// - For multi-objective [`Codomain`](crate::Codomain)s, `acc` is a
    ///   [`ParetoComputed`](crate::ParetoComputed) holding the current Pareto front.
    ///
    /// The accumulator is maintained externally by the [`Runable`](crate::Runable) and
    /// updated after each solution evaluation. The [`Optimizer`] should use it
    /// read-only to inform the generation of new candidates (e.g. for surrogate
    /// models or elitist strategies).
    fn step(
        &mut self,
        x: OptionCompShape<Scp, PSol, SolId, Self::SInfo, Self::Cod, Out>,
        scp: &Scp,
        acc: &CompAcc<Scp, PSol, SolId, Self::SInfo, Self::Cod, Out>,
    ) -> Scp::SolShape;
}

/// Multi-instance optimizer interface for parallel execution.
///
/// Implementations define how optimizer instances interact and synchronize.
pub trait MultiInstanceOptimizer<PSol, SolId, Opt, Out, Scp, Fn>:
    Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Self::SInfo>>,
{
    /// Exchange information with peer instances.
    fn interact(&self);
    /// Update internal state after interaction.
    fn update(&self);
}

/// Multi-fidelity marker trait for optimizers.
pub trait BudgetPruner<PSol, SolId, Opt, Out, Scp>: Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo> + HasFidelity + HasStep,
    SolId: Id,
    Opt: Domain,
    Out: FidOutcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
{
    /// Sets the minimum and maximum budgets for the optimizer.
    fn set_budgets(&mut self, budget_min: f64, budget_max: f64);

    /// Returns the current minimum and maximum budgets of the optimizer.
    fn get_budgets(&self) -> (f64, f64);

    /// Sets the current budget level used by the optimizer for pruning candidates.
    fn set_current_budget(&mut self, budget: f64);

    /// Get the current budget level used by the optimizer for pruning candidates.
    fn get_current_budget(&self) -> f64;

    /// Sets the scaling factor for budget levels in the optimizer.
    fn set_scaling(&mut self, scaling: f64);

    /// Returns the current scaling factor for budget levels in the optimizer.
    fn get_scaling(&self) -> f64;

    /// Drains all pending candidates from the optimizer, typically for cleanup.
    /// For example, in Hyperband, this can be used to clear all pending candidates
    /// when a new bracket is started, as they may not be relevant to the new budget configuration.
    fn drain(&mut self) -> Vec<Scp::SolShape>;

    /// Drains a single pending candidate from the optimizer, if available.
    fn drain_one(&mut self) -> Option<Scp::SolShape>;
}
