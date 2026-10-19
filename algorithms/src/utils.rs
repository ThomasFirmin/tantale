//! Utilities for basic algorithms implemented in Tantale algorithms

use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use tantale_core::{Dominate, Orderable, Problem, aliases::CompShape, err_msg, errors::ErrMsg};

pub mod mo;

pub enum PrunerError {
    BudgetError(&'static str),
    SamplerError(&'static str),
}
impl ErrMsg for PrunerError {
    fn _err_msg(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PrunerError::BudgetError(msg) => write!(f, "Pruner error: Invalid budget configuration. {msg}"),
            PrunerError::SamplerError(msg) => write!(f, "Pruner error: Sampler error. {msg}"),
        }
    }
}
err_msg!(PrunerError);

pub fn build_halving_budgets(min:f64, max:f64, scaling:f64) -> Result<Vec<f64>, PrunerError> {
    if scaling <= 1.0 {
        return Err(PrunerError::BudgetError("Scaling factor must be > 1.0"));
    }
    if min <= 0.0 {
        return Err(PrunerError::BudgetError("Minimum budget must be > 0.0"));
    }
    if max < min {
        return Err(PrunerError::BudgetError("Maximum budget must be > minimum budget"));
    }
    
    let mut budgets: Vec<f64> = (0..)
        .map(|i| min * scaling.powi(i))
        .take_while(|&b| b <= max)
        .collect();
    // If only one budget level is generated, add the max budget as a second level to ensure the algorithm can run
    if budgets.len() == 1 {
        budgets.push(max);
    }
    //If final budget is not budget_max, modify final budget to be budget_max
    if *budgets.last().unwrap() != max {
        let last = budgets.last_mut().unwrap();
        *last = max;
    }
    Ok(budgets)
}

/// A single entry in a rung of the Asynchronous Successive Halving algorithm.
#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "CompShape<P>: Serialize", deserialize = "CompShape<P>: Deserialize<'de>"))]
pub struct PromotableShape<P: Problem> {
    pub result: CompShape<P>,
    pub promoted: bool,
}

impl<P:Problem> Debug for PromotableShape<P> 
where
    CompShape<P>: Orderable,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromotableShape")
            .field("result", &self.result)
            .field("promoted", &self.promoted)
            .finish()
    }
}

impl<P:Problem> Orderable for PromotableShape<P>
where
    CompShape<P>: Orderable,
{
    fn ord_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.result.ord_cmp(&other.result)
    }
}

impl<P:Problem> Dominate for PromotableShape<P> 
where
    CompShape<P>: Dominate,
{
    fn dominates(&self, other: &Self) -> bool {
        self.result.dominates(&other.result)
    }
    
    fn get_objectives(&self) -> &[f64] {
        self.result.get_objectives()
    }
    
    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.result.get_objective_by_index(idx)
    }
    
    fn len_objectives(&self) -> usize {
        self.result.len_objectives()
    }
}

impl<P:Problem> Clone for PromotableShape<P> 
where
    CompShape<P>: Orderable,
{
    fn clone(&self) -> Self {
        PromotableShape {
            result: self.result.clone(),
            promoted: self.promoted,
        }
    }
}

impl<P:Problem> PromotableShape<P> 
where
    CompShape<P>: Orderable,
{
    /// Creates a new [`PromotableShape`] with the given result and sets the promoted flag to false.
    pub fn new(result: CompShape<P>) -> Self {
        PromotableShape {
            result,
            promoted: false,
        }
    }

    /// Compares the result of this [`PromotableShape`] with another, returning an ordering.
    pub fn ord_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.result.ord_cmp(&other.result)
    }

    /// Marks this [`PromotableShape`] as promoted.
    pub fn promote(&mut self) {
        self.promoted = true;
    }

    /// Checks if this [`PromotableShape`] has been promoted.
    pub fn is_promoted(&self) -> bool {
        self.promoted
    }
}

/// A type alias for a vector of vectors of [`PromotableShape`]s, 
/// representing the rungs in the Asynchronous Successive Halving algorithm
/// or its multi-objective version.
pub type Rungs<P> = Vec<Vec<PromotableShape<P>>>;