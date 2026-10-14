use tantale_core::{
    HasState, Problem, Sampled, Sampler, SolInfo, State, aliases::{Accum, CompShape, Shape}, optimizer::opt::{Batched, Prunable}, searchspace::Searchspace, utils::record::{Recordable, Schema},
};

use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use tantale_macros::basic_bounds;
use std::cell::RefCell;

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(rand::make_rng());
}

/// Metadata for a Random Search optimization step, associated to each [`Batch`].
#[derive(Serialize, Deserialize, Debug, Default)]
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

/// A helper macro to simplify the type signature of [`RandomSearch`].
#[macro_export]
macro_rules! randomsearch {
    () => {RandomSearch};
}


/// A helper macro to simplify the type signature of [`BatchRandomSearch`].
#[macro_export]
macro_rules! batchrandomsearch {
    () => {BatchRandomSearch<_>};
}

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
///
/// # Note
///
/// It implements [`SingleOptimizer`] for both [`BaseSol`] and [`FidelitySol`] solution types,
/// allowing it to handle [`Step`]-based optimization scenarios.
/// [`RandomSearch`] cannot [`Discard`](Step::Discard) any solutions, as it does not maintain
/// any state or history of evaluations.
/// All [`Partially`](Step::Partially) solutions will be re-outputed automatically, until [`Evaluated`](Step::Evaluated).
///
/// # Note on RNG
///
/// The optimizer uses a thread-local [`StdRng`] for random sampling.
/// The RNG is not part of the optimizer state, as it cannot be serialized or deserialized.
/// The [`StdRng`] is defined at the module level as follows:
/// ```rust,ignore
/// thread_local! {
///     static THREAD_RNG: RefCell<StdRng> = RefCell::new(rand::make_rng());
/// }
/// ```
/// It is called with a private function `with_rng` that takes a closure, allowing the optimizer to perform random sampling while keeping the RNG separate from the optimizer state:
/// ```rust,ignore
/// self.with_rng(|rng| scp.sample_opt(rng, EmptyInfo.into()))
/// ```
#[derive(Serialize, Deserialize, Debug)]
pub struct RandomSearch(pub usize);

impl RandomSearch {
    /// Creates a new instance of the Sequential [`RandomSearch`] optimizer with an initial state.
    pub fn new() -> Self {
        RandomSearch(0)
    }

    fn with_rng<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut StdRng) -> T,
    {
        THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
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
impl<P, Opt> Sampler<P, Opt> for RandomSearch
{
    type Info = RSInfo;

    fn sample_info(&mut self,scp: &P::Scp, _acc: &Accum<P>) -> Option<(Shape<P>, Sampled<RSInfo>)> {
        self.0 += 1;
        let shape = self.with_rng(|rng| scp.sample_pair(rng));
        Some((shape, RSInfo(self.0).into()))
    }

    fn update(&mut self, _x: &CompShape<P>, _scp: &P::Scp, _acc: &Accum<P>) { }
}

impl Prunable for RandomSearch {}

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "Shape<P>: Serialize + for<'a> Deserialize<'a>")]
pub struct BatchRandomSearch<P:Problem> 
{
    pub calls:usize,
    pub batch_size: usize,
    pub batch_uncomputed: Vec<Shape<P>>,
    pub batch_computed: Vec<Shape<P>>,
}

impl<P:Problem> BatchRandomSearch<P> {
    pub fn new(batch_size: usize) -> Self {
        BatchRandomSearch {
            calls: 0,
            batch_size,
            batch_uncomputed: Vec::with_capacity(batch_size),
            batch_computed: Vec::with_capacity(batch_size),
        }
    }

    fn with_rng<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut StdRng) -> T,
    {
        THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
    }
}

impl<P:Problem> State for BatchRandomSearch<P> {}
impl<P:Problem> HasState for BatchRandomSearch<P> {
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

    fn batch(&self) -> &[Shape<P>] {
        &self.batch_uncomputed
    }

    fn mut_batch(&mut self) -> &mut Vec<Shape<P>> {
        &mut self.batch_uncomputed
    }

    fn comp_batch(&self) -> &[CompShape<P>] {
        unreachable!("BatchRandomSearch does not maintain a batch of computed solutions");
    }

    fn mut_comp_batch(&mut self) -> &mut Vec<CompShape<P>> {
        unreachable!("BatchRandomSearch does not maintain a batch of computed solutions");
    }
}

#[basic_bounds(P, Opt)]
impl<P, Opt> Sampler<P, Opt> for BatchRandomSearch<P>
{
    type Info = RSInfo;

    fn sample_info(&mut self,scp: &P::Scp, _acc: &Accum<P>) -> Option<(Shape<P>, Sampled<RSInfo>)> {
        if self.batch_uncomputed.is_empty(){
            self.calls += 1;
            self.batch_uncomputed = self.with_rng(|rng| scp.vec_sample_pair(rng, self.batch_size));
            self.batch_computed.clear();
        }
        let shape = self.batch_uncomputed.pop()?;
        Some((shape, RSInfo(self.calls).into()))
    }

    fn update(&mut self, x: &CompShape<P>, _scp: &P::Scp, _acc: &Accum<P>) {
        self.add_to_comp_batch(x.clone());
    }
}
 
impl<P:Problem> Prunable for BatchRandomSearch<P> {}