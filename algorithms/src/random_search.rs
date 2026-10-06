use tantale_core::{
    Domain, FidOutcome, HasFidelity, HasState, HasStep, Problem, Sampler, SimpleProblem, SolId, SolInfo, State, Uncomputed, aliases::{Accum, CompShape, Shape}, objective::outcome::{FuncState, Outcome}, searchspace::Searchspace, utils::record::{Recordable, Schema},
};

use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
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
    type State = RandomSearch;

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

impl<P:SimpleProblem<Opt>, Opt: Domain> Sampler<P, Opt> for RandomSearch
{
    type Info = RSInfo;

    fn sample(&mut self,scp: &P::Scp,acc: &Accum<P>) -> Option<Shape<P>> {
        self.0 += 1;
        Some(self.with_rng(|rng| scp.sample_pair(rng, RSInfo(self.0).into())))
    }

    fn update(&mut self,x: CompShape<P>,scp: &P::Scp,acc: &Accum<P>) { }
}