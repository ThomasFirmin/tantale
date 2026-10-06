//! Utilities for basic algorithms implemented in Tantale algorithms

use tantale_core::{HasFidelity, err_msg, errors::ErrMsg};

pub mod mo;

/// A helper function to set the fidelity of a solution that implements the [`HasFidelity`] trait.
pub fn fidelity_setter<S: HasFidelity>(mut s: S, fidelity: f64) -> S {
    s.set_fidelity(fidelity);
    s
}

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
    if scaling < 1.0 {
        return Err(PrunerError::BudgetError("Scaling factor must be >= 1.0"));
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
