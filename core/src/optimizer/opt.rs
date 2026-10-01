use crate::{
    Problem, SolutionShape, aliases::{Accum, CompShape, Shape}, solution::SolInfo, 
    utils::{has_trait::HasState, record::{Recordable, Schema}},
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, ops::Fn};

/// Empty implementation for [`OptInfo`] or [`SolInfo`].
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct EmptyInfo;
impl Schema<()> for EmptyInfo {
    type Parts = ();
    fn columns(_context: &(), _part: Self::Parts) -> Vec<String> {
        Vec::new()
    }
}
impl Recordable<()> for EmptyInfo {
    type Parts = ();
    fn record(&self, _context: &(), _part: Self::Parts) -> Vec<String> {
        Vec::new()
    }
}
impl SolInfo for EmptyInfo {}

/// Core optimizer abstraction.
///
/// An [`Optimizer`] generates candidate [`Solution`](crate::solution::Solution)s to
/// **maximize** an objective, and maintains the state required to resume or
/// checkpoint an experiment.
pub trait Optimizer<P: Problem<Info = Self::Info>>: HasState
{
    type Info: SolInfo;
    
    /// Computes a single iteration of the [`Optimizer`].
    ///
    /// It consumes an optional computed solution and returns a new uncomputed
    /// candidate. [`Self`] is mutable to update the internal state.
    ///
    /// The `acc` parameter provides a read-only view of the best solutions
    /// accumulated since the start of the experiment:
    /// - For single-objective [`Codomain`](crate::Codomain)s, `acc` is a
    ///   [`BestAccumulator`](crate::BestAccumulator) holding the single best solution seen so far.
    /// - For multi-objective [`Codomain`](crate::Codomain)s, `acc` is a
    ///   [`ParetoAccumulator`](crate::ParetoAccumulator) holding the current Pareto front.
    ///
    /// The accumulator is maintained externally by the [`Runable`](crate::Runable) and
    /// updated after each solution evaluation. The [`Optimizer`] should use it
    /// read-only to inform the generation of new candidates (e.g. for surrogate
    /// models or elitist strategies).
    fn step(&mut self,x: Option<CompShape<P>>,scp: &P::Scp, acc: &Accum<P>) -> Option<Shape<P>>;
}

/// Multi-instance optimizer interface for parallel execution.
///
/// Implementations define how optimizer instances interact and synchronize.
pub trait MultiInstanceOptimizer<P: Problem<Info = Self::Info>>: Optimizer<P>
{
    /// Exchange information with peer instances.
    fn interact(&self);
    /// Update internal state after interaction.
    fn update(&self);
}

/// Multi-fidelity marker trait for optimizers.
pub trait BudgetPruner<P: Problem<Info = Self::Info>>: Optimizer<P>
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
    fn drain(&mut self) -> Vec<Shape<P>>;

    /// Drains a single pending candidate from the optimizer, if available.
    fn drain_one(&mut self) -> Option<Shape<P>>;
}

/// A [`Sampler`] is an [`Optimizer`] that always generates on-demand new candidates by sampling from an internal state.
/// Conversely to [`SingleOptimizer`]s or [`BatchOptimizer`]s,
/// a [`Sampler`] does not consume previously computed solutions to generate new candidates,
/// but only relies on its internal state.
pub trait Sampler<P, SuperSInfo> : HasState
where
    P: Problem,
    Shape<P>: SolutionShape<Info = Self::Info>,
    SuperSInfo: SolInfo,
{
    type Info: SolInfo;

    /// Generates a new candidate by sampling from the internal state of the [`Sampler`].
    fn sample(&mut self,scp: &P::Scp,acc: &Accum<P>) -> Option<Shape<P>>;

    fn sample_apply<F>(&mut self, f: F, scp: &P::Scp, acc: &Accum<P>) -> Option<Shape<P>>
    where
        F: Fn(Option<Shape<P>>) -> Option<Shape<P>> + Send + Sync
    {
        f(self.sample(scp, acc))
    }

    /// Update the internal state of the [`Sampler`] with a new [`Computed`](crate::Computed) candidate.
    fn update(&mut self,x: &CompShape<P>,scp: &P::Scp,acc: &Accum<P>);
}