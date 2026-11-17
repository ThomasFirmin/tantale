# Implementing a single-solution sampler

This guide follows the hill-climbing sampler in
`examples/examples/create_single_optimizer.rs`. It starts with a random
candidate; after an evaluation, it perturbs one coordinate of the current
best optimizer-side candidate. It assumes a nonempty [`Unit`](crate::core::Unit) optimizer domain
and a single-objective accumulator.

The runner asks a [`Sampler`](crate::core::Sampler) for one candidate and later calls `update` with
the computed result. In a parallel experiment, several candidates may be
outstanding and updates can arrive in a different order or several times
between sampling requests. "Single-solution" describes this sampling policy, not
a guarantee that an experiment runs one evaluation at a time.

## Metadata and checkpointable state

[`SolInfo`](crate::core::SolInfo) is metadata attached to each candidate, distinct from the sampler's
own state. Here the iteration is recorded using [`Schema`](crate::macros::Schema) and [`Recordable`](crate::macros::Recordable).
The sampler's iteration counter is itself the state. [`State`](crate::core::State) and [`HasState`](crate::core::HasState)
allow the runner to save and reconstruct it.

```rust
use serde::{Deserialize, Serialize};
use tantale::core::{HasState, SolInfo, State};
use tantale::macros::{Recordable, Schema};

#[derive(Schema, Recordable, Serialize, Deserialize, Debug, Default, Clone, Copy)]
pub struct HCInfo {
    pub iterations: usize,
}
impl SolInfo for HCInfo {}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct HillClimbing {
    iterations: usize,
}
impl State for HillClimbing {}
impl HasState for HillClimbing {
    type State = Self;
    fn get_state(&self) -> &Self { self }
    fn get_mut_state(&mut self) -> &mut Self { self }
    fn from_state(state: Self) -> Self { state }
}
```

## Proposing a candidate

The following implementation is adapted from the example. It uses
[`basic_bounds!`](crate::macros::basic_bounds) for the standard problem bounds and adds the requirements for
a single, orderable objective. The aliases `Shape<P>`, `CompShape<P>`, and
`Accum<P>` respectively denote an unevaluated candidate shape, a computed
shape, and the problem's accumulator.

Here we decide to specialize the sampler on a `Unit` domain. This optimizer-side domain, could be generic for
another sampler implementation.

```rust
# use serde::{Deserialize, Serialize};
# use tantale::core::{HasState, SolInfo, State};
# use tantale::macros::{Recordable, Schema};
# #[derive(Schema, Recordable, Serialize, Deserialize, Debug, Default, Clone, Copy)]
# pub struct HCInfo { pub iterations: usize }
# impl SolInfo for HCInfo {}
# #[derive(Serialize, Deserialize, Debug, Clone, Default)]
# pub struct HillClimbing { iterations: usize }
# impl State for HillClimbing {}
# impl HasState for HillClimbing {
#     type State = Self;
#     fn get_state(&self) -> &Self { self }
#     fn get_mut_state(&mut self) -> &mut Self { self }
#     fn from_state(state: Self) -> Self { state }
# }
use std::sync::Arc;
use rand::{RngExt, rngs::StdRng};
use tantale::{
    algos::utils::with_rng,
    basic_bounds,
    core::{
        HasInfo, HasX, Sampled, Sampler, Searchspace, SolutionShape, Unit,
        aliases::{Accum, CodElem, CompShape, Shape},
        domain::{ElemSingle, SingleAccumulator},
    },
};

fn neighbor(x: &[f64], rng: &mut StdRng) -> Arc<[f64]> {
    let mut next = x.to_vec();
    let i = rng.random_range(0..next.len());
    next[i] = (next[i] + rng.random_range(-0.1..0.1)).clamp(0.0, 1.0);
    Arc::from(next)
}

#[basic_bounds(P, Unit)]
impl<P> Sampler<P, Unit> for HillClimbing
where
    Accum<P>: SingleAccumulator<CompShape<P>>,
    CodElem<P>: ElemSingle + Ord,
{
    type Info = HCInfo;

    fn sample_info(&mut self, scp: &P::Scp, acc: &Accum<P>)
        -> Option<(Shape<P>, Sampled<HCInfo>)>
    {
        self.iterations += 1;
        let shape = match acc.get_best() {
            Some(best) => {
                let info = best.sinfo().find_info::<HCInfo>()
                    .expect("best candidate must carry hill-climbing metadata");
                println!("Best was sampled at iteration {}", info.iterations);
                let x = best.get_sopt().ref_x();
                scp.new_opt(with_rng(|rng| neighbor(x, rng)))
            }
            None => with_rng(|rng| scp.sample_pair(rng)),
        };
        Some((shape, HCInfo { iterations: self.iterations }.into()))
    }

    fn update(&mut self, _x: &CompShape<P>, _scp: &P::Scp, _acc: &Accum<P>) {
        // The next sample reads the updated accumulator.
    }
}
```

`sample_pair` draws a candidate when there is no best result. Once one exists,
`get_sopt().ref_x()` accesses its optimizer representation; `new_opt` creates
a shape from the perturbed optimizer value and maps it to the objective side.
`with_rng` provides the algorithm crate's thread-local RNG.

`info.into()` creates [`Sampled::Fresh`](crate::core::Sampled::Fresh) metadata. [`Sampled::Reused`](crate::core::Sampled::Reused) instead
keeps metadata already attached to a previously sampled shape, for example
when a multi-fidelity algorithm promotes a candidate. `find_info` locates
metadata even when another sampler wraps it. The example expects its metadata
to exist, so a missing entry is an error, not an ordinary first-sample case.

## Use in an experiment

Supply a problem whose optimizer domain is `Unit`, instantiate
`HillClimbing::default()`, and pass both to
`Experiment::new(problem, sampler, stop, recorder, checkpointer)`. The
problem's solution metadata type must match `HCInfo`; type inference normally
determines it from the sampler. For multi-fidelity implementations,
[`fidelity_bounds!`](crate::macros::fidelity_bounds) adds relevant fidelity bounds; a pruning algorithm also
needs its own promotion and budget logic.

The reference source is a compilable sampler example, but its `main` is empty;
it does not itself run an experiment.
