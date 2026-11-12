use std::sync::Arc;

use rand::{RngExt, rngs::StdRng};
use serde::{Deserialize, Serialize};
use tantale::{
    algos::utils::with_rng,
    basic_bounds,
    core::{
        HasInfo, HasState, HasX, HasY, Sampled, Sampler, Searchspace, SolInfo, SolutionShape,
        State, Unit,
        aliases::{Accum, CodElem, CompShape, Shape},
        domain::{ElemSingle, SingleAccumulator},
    },
    macros::{Recordable, Schema},
};

// A helper macro to simplify the type signature of `HillClimbing`.
// This a convention to make it easier to use the optimizer in generic contexts, without having to write out the full type signature.
// To use when generics can be infered in the usual load (resuming via a checkpoint) pipeline for example.
#[allow(unused_macros)]
macro_rules! hill_climbing {
    () => {
        HillClimbing
    };
}

/// Metadata for a Hill Climbing optimization step.
#[derive(Schema, Recordable, Serialize, Deserialize, Debug, Default, Clone, Copy)]
pub struct HCInfo {
    pub iterations: usize,
}

// Implement the SolInfo trait for HCInfo, allowing it to be used as metadata for solutions in the Hill Climbing algorithm.
impl SolInfo for HCInfo {}

/// Hill Climbing optimizer implementation.
/// Here the state IS the instance of the algorithm.
/// Sometimes it might be helpful to have a separate state struct, but in this case, the algorithm itself is simple enough to serve as its own state.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HillClimbing {
    iterations: usize,
}

impl HillClimbing {
    pub fn new() -> Self {
        Self { iterations: 0 }
    }
}

impl Default for HillClimbing {
    fn default() -> Self {
        Self::new()
    }
}

// Implement the State trait for HillClimbing. It allows checkpointing and restoring the state of the optimizer.
impl State for HillClimbing {}

// Implement the HasState trait for HillClimbing. Because HillClimbing is its own state, the methods simply return self.
impl HasState for HillClimbing {
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
impl<P> Sampler<P, Unit> for HillClimbing
where
    Accum<P>: SingleAccumulator<CompShape<P>>, // <= Constraint: The accumulator must support single-objective optimization.
    // CompShape<P> is the computed solution type (e.g. x and y)
    CodElem<P>: ElemSingle + Ord, // The element of the codomain (y) must be single-objective and orderable.
{
    type Info = HCInfo; // Metadata associated to each solution.

    // The 'sample_info' return a new candidate solution and its associated metadata.
    // If no more candidates can be generated, it returns None.
    fn sample_info(&mut self, scp: &P::Scp, acc: &Accum<P>) -> Option<(Shape<P>, Sampled<HCInfo>)> {
        self.iterations += 1;

        // Generate a new candidate near the current best solution.
        let shape = match acc.get_best() {
            Some(best) => {
                // Retrieve information about the best solution found so far.
                // We must use '.find_info' to access the BHCInfo metadata associated with the best solution.
                // This is because BHCInfo can be wrapped in other metadata types, and '.find_info' allows us to extract it regardless of the wrapping.
                // For example, a sampler might use an inner sampler that adds its own metadata, and '.find_info' will still allow us to access the BHCInfo.
                let info = best.sinfo().find_info::<HCInfo>().unwrap();
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
            // If there is no best solution yet, sample a random solution from the search space.
            None => with_rng(|rng| scp.sample_pair(rng)),
        };

        // Return the new candidate and its associated metadata.
        // If a solution is reused for example in the case of a multi-step evaluation,
        // then instead of returning a new metadate, one can return Sampled::Reused instead of metadata.into().
        Some((
            shape,
            HCInfo {
                iterations: self.iterations,
            }
            .into(),
        ))
    }

    // The 'update' method is called after a solution has been evaluated and added to the accumulator.
    // Be carefull update can be called multiple times before the next call to 'sample_info'.
    // For example, in the case of parallel computing.
    fn update(&mut self, _x: &CompShape<P>, _scp: &P::Scp, _acc: &Accum<P>) {
        // The next call to sample_info uses the updated best solution.
    }
}

fn main() {}
