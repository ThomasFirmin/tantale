use tantale_core::{
    HasState, Problem, Sampled, Sampler, SolInfo, State,
    aliases::{Accum, CompShape, Shape},
    optimizer::opt::{Batched, Prunable},
    searchspace::Searchspace,
    utils::{
        ShapeWithInfo,
        record::{Recordable, Schema},
    },
};

use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use tantale_macros::basic_bounds;

use crate::utils::{refill_apply, with_rng};

/// A helper macro to simplify the type signature of [`RandomSearch`].
#[macro_export]
macro_rules! randomsearch {
    () => {
        RandomSearch
    };
}

/// A helper macro to simplify the type signature of [`BatchRandomSearch`].
#[macro_export]
macro_rules! batchrandomsearch {
    () => {BatchRandomSearch<_>};
}

/// Metadata for a Random Search optimization step, associated to each [`Batch`].
#[derive(Serialize, Deserialize, Debug, Default, Clone, Copy)]
pub struct RSInfo(pub usize);

impl Schema<()> for RSInfo {
    type Parts = ();
    fn columns(_context: &(), _part: Self::Parts) -> Vec<String> {
        vec!["calls".to_string()]
    }
}
impl Recordable<()> for RSInfo {
    type Parts = ();
    fn record(&self, _context: &(), _part: Self::Parts) -> Vec<String> {
        vec![self.0.to_string()]
    }
}
impl SolInfo for RSInfo {}

//------------------//
//--- SEQUENTIAL ---//
//------------------//

/// Random Search sampler implementation.
/// This algorithm radomly samples solutions on-demand
/// from the [`Searchspace`] (using [`sample_pair`](Searchspace::sample_pair)) at each iteration, without any internal state or memory of past evaluations.
///
/// # Workflow
///
/// ```text
///  Worker requests solution
///           |
///           v
///  +--------------------+
///  | Prior solution     |
///  | provided?          |
///  +--------------------+
///     Yes /         | No
///        /          |
///       v           |
///  +----------+     |
///  | Check    |     +----------------+
///  | status   |                      |
///  +----------+                      |
///       |                            |
///       v                            v
///    +---------+           +-------------------+
///   / Partially \          | Sample new random |  
///  /   stepped?  \  No --> | solution from     |
///  \             /         | searchspace       |
///   \   Yes     /           +-------------------+  
///    +---------+                     |      
///        |                           |
///        v                           v
///  Return same solution      Return the solution
///  (continue evaluation)
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RandomSearch(pub usize);

impl RandomSearch {
    /// Creates a new instance of the Sequential [`RandomSearch`] optimizer with an initial state.
    pub fn new() -> Self {
        RandomSearch(0)
    }
}

impl Default for RandomSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl State for RandomSearch {}

impl HasState for RandomSearch {
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

#[basic_bounds(P, Opt)]
impl<P, Opt> Sampler<P, Opt> for RandomSearch {
    type Info = RSInfo;

    fn sample_info(
        &mut self,
        scp: &P::Scp,
        _acc: &Accum<P>,
    ) -> Option<(Shape<P>, Sampled<RSInfo>)> {
        self.0 += 1;
        let shape = with_rng(|rng| scp.sample_pair(rng));
        Some((shape, RSInfo(self.0).into()))
    }

    fn update(&mut self, _x: &CompShape<P>, _scp: &P::Scp, _acc: &Accum<P>) {}
}

impl Prunable for RandomSearch {}

#[derive(Serialize, Deserialize)]
#[serde(bound = "ShapeWithInfo<Shape<P>, RSInfo>: Serialize + for<'a> Deserialize<'a>")]
pub struct BatchRandomSearch<P: Problem> {
    pub calls: usize,
    pub batch_size: usize,
    pub batch_uncomputed: Vec<ShapeWithInfo<Shape<P>, RSInfo>>,
    pub batch_computed: Vec<CompShape<P>>,
}

impl<P: Problem> Clone for BatchRandomSearch<P> {
    fn clone(&self) -> Self {
        BatchRandomSearch {
            calls: self.calls,
            batch_size: self.batch_size,
            batch_uncomputed: self.batch_uncomputed.clone(),
            batch_computed: self.batch_computed.clone(),
        }
    }
}

impl<P: Problem> Debug for BatchRandomSearch<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchRandomSearch")
            .field("calls", &self.calls)
            .field("batch_size", &self.batch_size)
            .field("batch_uncomputed", &self.batch_uncomputed)
            .field("batch_computed", &self.batch_computed)
            .finish()
    }
}

impl<P: Problem> BatchRandomSearch<P> {
    pub fn new(batch_size: usize) -> Self {
        BatchRandomSearch {
            calls: 0,
            batch_size,
            batch_uncomputed: Vec::with_capacity(batch_size),
            batch_computed: Vec::with_capacity(batch_size),
        }
    }
}

impl<P: Problem> State for BatchRandomSearch<P> {}
impl<P: Problem> HasState for BatchRandomSearch<P> {
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

#[basic_bounds(P, Opt)]
impl<P, Opt> Batched<P, Opt> for BatchRandomSearch<P> {
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn set_batch_size(&mut self, size: usize) {
        self.batch_size = size;
    }

    fn batch(&self) -> &[ShapeWithInfo<Shape<P>, RSInfo>] {
        &self.batch_uncomputed
    }

    fn mut_batch(&mut self) -> &mut Vec<ShapeWithInfo<Shape<P>, RSInfo>> {
        &mut self.batch_uncomputed
    }

    fn comp_batch(&self) -> &[CompShape<P>] {
        self.batch_computed.as_slice()
    }

    fn mut_comp_batch(&mut self) -> &mut Vec<CompShape<P>> {
        &mut self.batch_computed
    }
}

#[basic_bounds(P, Opt)]
impl<P, Opt> Sampler<P, Opt> for BatchRandomSearch<P> {
    type Info = RSInfo;

    fn sample_info(
        &mut self,
        scp: &P::Scp,
        _acc: &Accum<P>,
    ) -> Option<(Shape<P>, Sampled<RSInfo>)> {
        if self.batch_uncomputed.is_empty() {
            self.calls += 1;
            refill_apply::<P, RSInfo, _>(&mut self.batch_uncomputed, scp, self.batch_size, |s| {
                (s, RSInfo(self.calls))
            });
            self.batch_computed.clear();
        }
        let shape = self.batch_uncomputed.pop()?;
        Some((shape.0, shape.1))
    }

    fn update(&mut self, x: &CompShape<P>, _scp: &P::Scp, _acc: &Accum<P>) {
        self.add_to_comp_batch(x.clone());
    }
}

impl<P: Problem> Prunable for BatchRandomSearch<P> {}
