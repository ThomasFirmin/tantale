use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use rand::{RngExt, rngs::StdRng};
use serde::{Deserialize, Serialize};
use tantale::{
    algos::utils::with_rng,
    basic_bounds,
    core::{
        HasInfo, HasState, HasX, HasY, Problem, Sampled, Sampler, Searchspace, SolInfo,
        SolutionShape, State, Unit,
        aliases::{Accum, CodElem, CompShape, Shape},
        domain::{ElemSingle, SingleAccumulator},
        utils::ShapeWithInfo,
    },
    macros::{Recordable, Schema},
};

// A helper macro to simplify the type signature of `BatchedHillClimbing`.
// This a convention to make it easier to use the optimizer in generic contexts, without having to write out the full type signature.
// To use when generics can be infered in the usual load (resuming via a checkpoint) pipeline for example.
#[allow(unused_macros)]
macro_rules! batched_hill_climbing {
    () => {
        BatchedHillClimbing
    };
}

/// Metadata for a Batched Hill Climbing optimization step.
#[derive(Schema, Recordable, Serialize, Deserialize, Debug, Default, Clone, Copy)]
pub struct BHCInfo {
    pub iterations: usize,
}

// Implement the SolInfo trait for BHCInfo, allowing it to be used as metadata for solutions in the Hill Climbing algorithm.
impl SolInfo for BHCInfo {}

impl BHCInfo {
    pub fn new(iterations: usize) -> Self {
        Self { iterations }
    }
}

impl Display for BHCInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Iteration : {}", self.iterations)
    }
}

/// Batched Hill Climbing optimizer implementation.
/// Here the state IS the instance of the algorithm.
/// Sometimes it might be helpful to have a separate state struct, but in this case, the algorithm itself is simple enough to serve as its own state.
#[derive(Serialize, Deserialize)]
#[serde(bound = "ShapeWithInfo<Shape<P>, BHCInfo>: Serialize + for<'a> Deserialize<'a>")]
pub struct BatchedHillClimbing<P: Problem> {
    iterations: usize,
    batch_size: usize, // The number of candidate solutions to generate in each iteration.
    // We use ShapeWithInfo<Shape<P>, BHCInfo> to store both the candidate solution and its associated metadata (BHCInfo).
    batch: Vec<ShapeWithInfo<Shape<P>, BHCInfo>>, // The batch of candidate solutions generated in the current iteration.
    // We do not use ShapeWithInfo<CompShape<P>, BHCInfo> because the inner pipeline automatically assigns the BHCInfo
    // to the computed solutions. The metadata can be accessed via '.sinfo' (HasInfo trait)
    comp_batch: Vec<CompShape<P>>, // The batch of computed solutions (e.g., x and y) corresponding to the candidate solutions.
    pending: usize,                // The number of pending evaluations in the current batch.
}

// Implement Clone and Debug by hand for BatchedHillClimbing.
// This is necessary because the generic type P may not implement Clone or Debug, and we want to avoid requiring those traits on P.
impl<P: Problem> Clone for BatchedHillClimbing<P> {
    fn clone(&self) -> Self {
        Self {
            iterations: self.iterations,
            batch_size: self.batch_size,
            batch: self.batch.clone(),
            comp_batch: self.comp_batch.clone(),
            pending: self.pending,
        }
    }
}

impl<P: Problem> Debug for BatchedHillClimbing<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedHillClimbing")
            .field("iterations", &self.iterations)
            .field("batch_size", &self.batch_size)
            .field("batch", &self.batch)
            .field("comp_batch", &self.comp_batch)
            .field("pending", &self.pending)
            .finish()
    }
}

impl<P: Problem> BatchedHillClimbing<P> {
    pub fn new(batch_size: usize) -> Self {
        Self {
            iterations: 0,
            batch_size,
            batch: Vec::new(),
            comp_batch: Vec::new(),
            pending: 0,
        }
    }
}

// Implement the State trait for BatchedHillClimbing. It allows checkpointing and restoring the state of the optimizer.
impl<P: Problem> State for BatchedHillClimbing<P> {}

// Implement the HasState trait for BatchedHillClimbing. Because BatchedHillClimbing is its own state, the methods simply return self.
impl<P: Problem> HasState for BatchedHillClimbing<P> {
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

/// A simple neighborhood function, working in [0., 1.], that perturbs a solution by adding a small random value to one of its dimensions.
fn neighbor(x: &[f64], rng: &mut StdRng) -> Arc<[f64]> {
    let mut new_x = x.to_vec();
    let idx = rng.random_range(0..new_x.len());
    let perturbation: f64 = rng.random_range(-0.1..0.1);
    new_x[idx] = (new_x[idx] + perturbation).clamp(0., 1.);
    Arc::from(new_x)
}

// the basic_bounds macro is used to add basic trait bounds to a sampler for an HPO problem.
// It ensures that the sampler can work with any problem P that has a Unit domain.
// The macro expands to the necessary trait bounds for the sampler to be compatible with the HPO framework.
// The domain can be generic.
#[basic_bounds(P, Unit)] // <= Constraint: The optimizer works with any problem P within a Unit domain.
impl<P> Sampler<P, Unit> for BatchedHillClimbing<P>
where
    Accum<P>: SingleAccumulator<CompShape<P>>, // <= Constraint: The accumulator must support single-objective optimization.
    // CompShape<P> is the computed solution type (e.g. x and y)
    CodElem<P>: ElemSingle + Ord, // The element of the codomain (y) must be single-objective and orderable.
{
    type Info = BHCInfo; // Metadata associated to each solution.

    // The 'sample_info' return a new candidate solution and its associated metadata.
    // If no more candidates can be generated, it returns None.
    fn sample_info(
        &mut self,
        scp: &P::Scp,
        acc: &Accum<P>,
    ) -> Option<(Shape<P>, Sampled<BHCInfo>)> {
        // If there are still uncomputed solutions in the current batch, return one of them.
        if let Some(sol) = self.batch.pop() {
            self.pending += 1;
            Some(sol.into())
        } else if self.pending > 0 {
            // If there are still pending evaluations, return None to wait for them to complete.
            None
        } else {
            // If there are no pending evaluations and no uncomputed solutions, generate a new batch of candidate solutions.
            self.iterations += 1;
            // Generate a new batch of candidate solutions.
            self.batch = (0..self.batch_size)
                .map(|_| {
                    let sol = match acc.get_best() {
                        Some(best) => {
                            // Retrieve information about the best solution found so far.
                            // We must use '.find_info' to access the BHCInfo metadata associated with the best solution.
                            // This is because BHCInfo can be wrapped in other metadata types, and '.find_info' allows us to extract it regardless of the wrapping.
                            // For example, a sampler might use an inner sampler that adds its own metadata, and '.find_info' will still allow us to access the BHCInfo.
                            let info = best.sinfo().find_info::<BHCInfo>().unwrap();
                            // Work with Optimizer side solution and retrieve 'x' the raw solution.
                            let x = best.get_sopt().ref_x();
                            let y = best.y().value();
                            println!(
                                "Best solution found so far: f({:?}) ={} at iteration {}",
                                x, y, info.iterations
                            );
                            // Generate a new candidate in the neighborhood of the best solution.
                            // Use 'with_rng' to ensure that the random number generator is properly seeded and used. (e.g. for multi-threaded scenarios)
                            let x = with_rng(|rng| neighbor(x, rng));
                            scp.new_opt(x)
                        }
                        None => with_rng(|rng| scp.sample_pair(rng)),
                    };
                    let info = Sampled::Fresh(BHCInfo::new(self.iterations).into());
                    ShapeWithInfo(sol, info)
                })
                .collect();
            self.sample_info(scp, acc)
        }
    }

    // The 'update' method is called after a solution has been evaluated and added to the accumulator.
    // Be carefull update can be called multiple times before the next call to 'sample_info'.
    // For example, in the case of parallel computing.
    fn update(&mut self, _x: &CompShape<P>, _scp: &P::Scp, _acc: &Accum<P>) {
        if self.pending > 0 {
            self.pending -= 1;
        } else {
            println!(
                "Warning: update called with no pending evaluations. This may indicate a logic error in the optimizer."
            );
        }
    }
}

fn main() {}
