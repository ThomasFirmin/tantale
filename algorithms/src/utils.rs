//! Utilities for basic algorithms implemented in Tantale algorithms

use std::{cell::RefCell, fmt::Debug};

use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use tantale_core::{
    Dominate, Orderable, Problem, Recordable, Schema, Searchspace, SolInfo,
    aliases::{CompShape, Shape},
    err_msg,
    errors::ErrMsg,
    utils::ShapeWithInfo,
};

pub mod mo;

pub enum PrunerError {
    BudgetError(&'static str),
    SamplerError(&'static str),
}
impl ErrMsg for PrunerError {
    fn _err_msg(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PrunerError::BudgetError(msg) => {
                write!(f, "Pruner error: Invalid budget configuration. {msg}")
            }
            PrunerError::SamplerError(msg) => write!(f, "Pruner error: Sampler error. {msg}"),
        }
    }
}
err_msg!(PrunerError);

pub fn build_halving_budgets(min: f64, max: f64, scaling: f64) -> Result<Vec<f64>, PrunerError> {
    if !scaling.is_finite() || scaling <= 1.0 {
        return Err(PrunerError::BudgetError(
            "Scaling factor must be finite and > 1.0",
        ));
    }
    if !min.is_finite() || min <= 0.0 {
        return Err(PrunerError::BudgetError(
            "Minimum budget must be finite and > 0.0",
        ));
    }
    if !max.is_finite() || max < min {
        return Err(PrunerError::BudgetError(
            "Maximum budget must be finite and >= minimum budget",
        ));
    }

    if min == max {
        return Ok(vec![max]);
    }

    let mut budgets = vec![min];

    loop {
        let current = *budgets.last().unwrap();
        let next = current * scaling;

        if next <= current {
            return Err(PrunerError::BudgetError(
                "Scaling does not produce a larger representable budget",
            ));
        }

        if !next.is_finite() || next >= max {
            break;
        }

        budgets.push(next);
    }

    // Always include max exactly once.
    budgets.push(max);

    Ok(budgets)
}

pub fn build_halving_budgets_with_levels(
    min: f64,
    max: f64,
    scaling: f64,
    num_levels: usize,
) -> Result<Vec<f64>, PrunerError> {
    if !scaling.is_finite() || scaling <= 1.0 {
        return Err(PrunerError::BudgetError(
            "Scaling factor must be finite and > 1.0",
        ));
    }

    if !min.is_finite() || min <= 0.0 {
        return Err(PrunerError::BudgetError(
            "Minimum budget must be finite and > 0.0",
        ));
    }

    if !max.is_finite() || max < min {
        return Err(PrunerError::BudgetError(
            "Maximum budget must be finite and >= minimum budget",
        ));
    }

    if num_levels == 0 {
        return Err(PrunerError::BudgetError("Number of levels must be > 0"));
    }

    if num_levels == 1 {
        if min != max {
            return Err(PrunerError::BudgetError(
                "A single level requires min == max",
            ));
        }

        return Ok(vec![max]);
    }

    let mut budgets = Vec::with_capacity(num_levels);
    budgets.push(min);

    for _ in 1..num_levels - 1 {
        let current = *budgets.last().unwrap();
        let next = current * scaling;

        if !next.is_finite() || next <= current {
            return Err(PrunerError::BudgetError(
                "Invalid or non-increasing budget sequence",
            ));
        }

        if next >= max {
            return Err(PrunerError::BudgetError(
                "Maximum budget is too small for the requested number of levels",
            ));
        }

        budgets.push(next);
    }

    budgets.push(max);

    Ok(budgets)
}

/// A single entry in a rung of the Asynchronous Successive Halving algorithm.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "CompShape<P>: Serialize",
    deserialize = "CompShape<P>: Deserialize<'de>"
))]
pub struct PromotableShape<P: Problem> {
    pub result: CompShape<P>,
    pub promoted: bool,
}

impl<P: Problem> Debug for PromotableShape<P>
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

impl<P: Problem> Orderable for PromotableShape<P>
where
    CompShape<P>: Orderable,
{
    fn ord_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.result.ord_cmp(&other.result)
    }
}

impl<P: Problem> Dominate for PromotableShape<P>
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

impl<P: Problem> Clone for PromotableShape<P>
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

impl<P: Problem> PromotableShape<P>
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BracketInfo(pub usize);

impl Schema<()> for BracketInfo {
    type Parts = ();
    fn columns(_context: &(), _part: Self::Parts) -> Vec<String> {
        vec!["bracket".into()]
    }
}

impl Recordable<()> for BracketInfo {
    type Parts = ();
    fn record(&self, _context: &(), _part: Self::Parts) -> Vec<String> {
        vec![self.0.to_string()]
    }
}

impl SolInfo for BracketInfo {}

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(rand::make_rng());
}

/// Executes a closure with a thread-local random number generator.
///
/// # Example
///
/// ```rust
/// use rand::RngExt;
/// use tantale_algos::utils::with_rng;
///
/// let random_value = with_rng(|rng| rng.random::<f64>());
/// assert!((0.0..1.0).contains(&random_value));
/// ```
pub fn with_rng<F, T>(f: F) -> T
where
    F: FnOnce(&mut StdRng) -> T,
{
    THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
}

/// Refills a vector of [`Shape`]s until it reaches the specified length by sampling from the provided [`Searchspace`].
pub fn refill<P, I>(v: &mut Vec<ShapeWithInfo<Shape<P>, I>>, scp: &P::Scp, until: usize)
where
    P: Problem,
    I: SolInfo,
{
    while v.len() < until {
        let sol = with_rng(|rng| scp.sample_pair(rng));
        v.push((sol, I::default()).into());
    }
}

/// Refills a vector of [`Shape`]s until it reaches the specified length by sampling from the provided [`Searchspace`],
/// and applies a transformation function to each sampled shape.
pub fn refill_apply<P, I, F>(
    v: &mut Vec<ShapeWithInfo<Shape<P>, I>>,
    scp: &P::Scp,
    until: usize,
    f: F,
) where
    P: Problem,
    I: SolInfo,
    F: Fn(Shape<P>) -> (Shape<P>, I) + Send + Sync,
{
    while v.len() < until {
        let res = f(with_rng(|rng| scp.sample_pair(rng)));
        v.push(res.into())
    }
}
