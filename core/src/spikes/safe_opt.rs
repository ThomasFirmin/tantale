use std::cmp::Ordering;

use crate::{CompShape, Domain, FidOutcome, HasFidelity, HasStep, Id, Optimizer, Outcome, Sampler, Searchspace, Uncomputed, optimizer::opt::BudgetPruner};

/// A trait for optimizers that implement safe pruning based on fidelity and spikes. 
/// This trait extends the [`Optimizer`] trait 
/// and provides additional methods for managing safe budgets and pruning candidates.
pub trait SafePruner<PSol, SolId, Opt, Out, Scp>: Optimizer<PSol, SolId, Opt, Out, Scp>
where
    Self: BudgetPruner<PSol, SolId, Opt, Out, Scp>,
    PSol: Uncomputed<SolId, Opt, Self::SInfo> + HasFidelity + HasStep,
    PSol::Twin<Scp::Obj>:
        Uncomputed<SolId, Scp::Obj, Self::SInfo, Twin<Opt> = PSol> + HasFidelity + HasStep,
    SolId: Id,
    Opt: Domain,
    Out: FidOutcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
{
    /// Sets the minimum and maximum safe budgets for the optimizer.
    fn set_safe_budgets(&mut self, budget_min: f64, budget_max: f64);

    /// Returns the current minimum and maximum safe budgets of the optimizer.
    fn get_safe_budgets(&self) -> (f64, f64);

    /// Sets the current safe budget level used by the optimizer for pruning candidates.
    fn set_current_safe_budget(&mut self, budget: f64);

    /// Get the current safe budget level used by the optimizer for pruning candidates.
    fn get_current_safe_budget(&self) -> f64;

    /// Sets the safe scaling factor for budget levels in the optimizer.
    fn set_safe_scaling(&mut self, scaling: f64);

    /// Returns the safe current scaling factor for budget levels in the optimizer.
    fn get_safe_scaling(&self) -> f64;
}


/// A [`Sampler`] is an [`Optimizer`] that always generates on-demand new candidates by sampling from an internal state.
/// Conversely to [`SingleOptimizer`](crate::SingleOptimizer)s or [`BatchOptimizer`](crate::BatchOptimizer)s,
/// a [`Sampler`] does not consume previously computed solutions to generate new candidates,
/// but only relies on its internal state.
pub trait SafeSampler<PSol, SolId, Opt, Out, Scp>: Sampler<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Self::SInfo, Twin<Opt> = PSol>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
{ 
    /// Computes the safety metric for a given candidate solution.
    fn safety(&self, candidate: &CompShape<Scp::SolShape, SolId, Self::SInfo, Out>) -> f64;

    /// Compares two candidate solutions based on their safety metrics.
    fn comp_safety(&self, a: &CompShape<Scp::SolShape, SolId, Self::SInfo, Out>, b: &CompShape<Scp::SolShape, SolId, Self::SInfo, Out>) -> Ordering;

    /// Computes the utility metric for a given candidate solution, which may include performance and safety.
    fn utility(&self, candidate: &CompShape<Scp::SolShape, SolId, Self::SInfo, Out>) -> f64;

    /// Compares two candidate solutions based on their utility metrics, which may include performance and safety.
    fn comp_utility(&self, a: &CompShape<Scp::SolShape, SolId, Self::SInfo, Out>, b: &CompShape<Scp::SolShape, SolId, Self::SInfo, Out>) -> Ordering;

    /// Sets the safe and regular budgets for the optimizer.
    /// Used to manage the trade-off between exploration and exploitation in the multi-fidelity optimization process
    fn set_budgets(&mut self, safe_budgets: (f64, f64), regular_budgets: (f64, f64)) -> Result<(), String>;
}