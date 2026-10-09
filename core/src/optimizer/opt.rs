use crate::{
    Domain, HasFidelity, HasStep, Problem, aliases::{Accum, CompShape, Opt, Shape}, solution::SolInfo, utils::{has_trait::HasState, record::{Recordable, Schema}},
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
/// A [`Sampler`] generates candidate [`Solution`](crate::solution::Solution)s to
/// **maximize** an objective, and maintains the state required to resume or
/// checkpoint an experiment.
pub trait Sampler<P: Problem<Info = Self::Info>, Dom: Domain>: HasState
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

    /// Update the internal state of the [`Sampler`] with a new [`Computed`](crate::Computed) candidate and sample a new candidate.
    fn update_and_sample(&mut self, x: &CompShape<P>, scp: &P::Scp, acc: &Accum<P>) -> Option<Shape<P>> {
        self.update(x, scp, acc);
        self.sample(scp, acc)
    }

    /// Update the internal state of the [`Sampler`] with a new [`Computed`](crate::Computed) candidate and sample a new candidate, 
    /// applying a function to the result.
    fn update_and_sample_apply<F>(&mut self, x: &CompShape<P>, f: F, scp: &P::Scp, acc: &Accum<P>) -> Option<Shape<P>>
    where
        F: Fn(Option<Shape<P>>) -> Option<Shape<P>> + Send + Sync
    {
        self.update(x, scp, acc);
        f(self.sample(scp, acc))
    }
}

pub trait Batched<P: Problem<Info = Self::Info>, Dom: Domain>: Sampler<P, Dom>
{
    /// Returns the current batch size used by the optimizer for sampling candidates.
    fn batch_size(&self) -> usize;
    
    /// Sets the batch size used by the optimizer for sampling candidates.
    fn set_batch_size(&mut self, size: usize);
    
    /// Returns the current batch of candidates.
    fn batch(&self) -> &[Shape<P>];
    
    /// Returns a mutable reference to the current batch of candidates.
    fn mut_batch(&mut self) -> &mut Vec<Shape<P>>;
    
    /// Returns the current batch of computed candidates.
    fn comp_batch(&self) -> &[CompShape<P>];
    
    /// Returns a mutable reference to the current batch of computed candidates.
    fn mut_comp_batch(&mut self) -> &mut Vec<CompShape<P>>;

    /// Adds a candidate to the current batch.
    fn add_to_batch(&mut self, candidate: Shape<P>) {
        self.mut_batch().push(candidate);
    }

    /// Pops a candidate from the current batch.
    fn pop_from_batch(&mut self) -> Option<Shape<P>> {
        self.mut_batch().pop()
    }

    /// Adds a computed candidate to the current batch of computed candidates.
    fn add_to_comp_batch(&mut self, candidate: CompShape<P>) {
        self.mut_comp_batch().push(candidate);
    }

    /// Pops a computed candidate from the current batch of computed candidates.
    fn pop_from_comp_batch(&mut self) -> Option<CompShape<P>> {
        self.mut_comp_batch().pop()
    }
}

/// Multi-instance optimizer interface for parallel execution.
///
/// Implementations define how optimizer instances interact and synchronize.
pub trait MultiInstanceSampler<P: Problem<Info = Self::Info>>: Sampler<P, Opt<P>>
{
    /// Exchange information with peer instances.
    fn interact(&self);
    /// Update internal state after interaction.
    fn update(&self);
}

/// Multi-fidelity marker trait for optimizers.
pub trait Pruner<P: Problem<Info = Self::Info>, Opt: Domain>: Sampler<P, Opt>
where
    P::Shape: HasFidelity + HasStep,
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