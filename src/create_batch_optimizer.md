# Implementing a batched sampler

This guide follows the batched hill-climbing sampler in
`examples/examples/create_batch_optimizer.rs`. It generates a group
(batch) of candidates around the current best optimizer-side candidate,
returns them out one at a time, and waits until every candidate of the batch has
been evaluated before generating the next batch. It assumes a [`Unit`](crate::core::Unit)
optimizer domain and a single-objective accumulator, like the
[sequential sampler](crate::docs::CreateSingleSolutionSampler).

The runner only ever calls [`Sampler::sample_info`](crate::core::Sampler::sample_info) (one candidate per call)
and [`Sampler::update`](crate::core::Sampler::update) (one evaluated candidate per call). A batched sampler is
therefore a sampler that keeps an internal queue: it fills the queue, pops
candidates from it, and uses `None` to tell the runner that no candidate is
available yet. In a parallel experiment, workers that find no candidate
simply wait; this is how a batch acts as a synchronization point.

## Metadata and checkpointable state

As in the sequential guide, [`SolInfo`](crate::core::SolInfo) metadata is attached to each candidate
and the sampler's own state is checkpointed through [`State`](crate::core::State) and [`HasState`](crate::core::HasState).
The difference is that the state now contains candidates:

- `iterations`: number of batches generated so far.
- `batch_size`: number of candidates per batch.
- `batch`: candidates generated but not yet evaluted, each paired with its
  metadata (`ShapeWithInfo<Shape<P>, BHCInfo>`).
- `pending`: returned candidates whose evaluation has not been reported
  through `update`.
- `comp_batch`: a field intended for computed candidates. The example never
  fills or reads it, because the metadata and results are already available
  through the accumulator.

Because the state depends on the problem type `P`, the struct is generic and
needs two adjustments. The Serde bound states that the stored shape must be
serializable, and `Clone` and `Debug` are written by hand so that `P` itself
does not need to implement them.

```rust
use std::fmt::Debug;
use serde::{Deserialize, Serialize};
use tantale::core::{
    HasState, Problem, SolInfo, State,
    aliases::{CompShape, Shape},
    utils::ShapeWithInfo,
};
use tantale::macros::{Recordable, Schema};

#[derive(Schema, Recordable, Serialize, Deserialize, Debug, Default, Clone, Copy)]
pub struct BHCInfo {
    pub iterations: usize,
}
impl SolInfo for BHCInfo {}

#[derive(Serialize, Deserialize)]
#[serde(bound = "ShapeWithInfo<Shape<P>, BHCInfo>: Serialize + for<'a> Deserialize<'a>")]
pub struct BatchedHillClimbing<P: Problem> {
    iterations: usize,
    batch_size: usize,
    batch: Vec<ShapeWithInfo<Shape<P>, BHCInfo>>,
    comp_batch: Vec<CompShape<P>>,
    pending: usize,
}

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

impl<P: Problem> State for BatchedHillClimbing<P> {}
impl<P: Problem> HasState for BatchedHillClimbing<P> {
    type State = Self;
    fn get_state(&self) -> &Self { self }
    fn get_mut_state(&mut self) -> &mut Self { self }
    fn from_state(state: Self) -> Self { state }
}
```

Serializing the queue matters: when an experiment is resumed from a
checkpoint, candidates already generated are served again instead of being
lost, and `pending` keeps the batch synchronization consistent.

## Serving candidates and building batches

The [`Sampler`](crate::core::Sampler) implementation has three cases in `sample_info`, checked in
this order:

1. The queue is not empty: pop a candidate, count it as pending, return it.
   Its metadata was created when the batch was generated, so it is returned
   as is ([`ShapeWithInfo`](crate::core::utils::ShapeWithInfo) converts into the expected return value).
2. The queue is empty but `pending > 0`: return `None`. The previous batch is
   still being evaluated.
3. The queue is empty and nothing is pending: build a new batch from the
   current best result, then serve from it by calling `sample_info` again.

`update` decrements `pending` for each reported evaluation. Updates can arrive
in any order and in groups, so only the count matters.

```rust
# use std::{fmt::Debug, sync::Arc};
# use rand::{RngExt, rngs::StdRng};
# use serde::{Deserialize, Serialize};
# use tantale::core::{
#     HasState, Problem, SolInfo, State,
#     aliases::{CompShape, Shape},
#     utils::ShapeWithInfo,
# };
# use tantale::macros::{Recordable, Schema};
# #[derive(Schema, Recordable, Serialize, Deserialize, Debug, Default, Clone, Copy)]
# pub struct BHCInfo { pub iterations: usize }
# impl SolInfo for BHCInfo {}
# impl BHCInfo { pub fn new(iterations: usize) -> Self { Self { iterations } } }
# #[derive(Serialize, Deserialize)]
# #[serde(bound = "ShapeWithInfo<Shape<P>, BHCInfo>: Serialize + for<'a> Deserialize<'a>")]
# pub struct BatchedHillClimbing<P: Problem> {
#     iterations: usize,
#     batch_size: usize,
#     batch: Vec<ShapeWithInfo<Shape<P>, BHCInfo>>,
#     comp_batch: Vec<CompShape<P>>,
#     pending: usize,
# }
# impl<P: Problem> Clone for BatchedHillClimbing<P> {
#     fn clone(&self) -> Self {
#         Self { iterations: self.iterations, batch_size: self.batch_size,
#                batch: self.batch.clone(), comp_batch: self.comp_batch.clone(),
#                pending: self.pending }
#     }
# }
# impl<P: Problem> Debug for BatchedHillClimbing<P> {
#     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
#         f.debug_struct("BatchedHillClimbing").finish()
#     }
# }
# impl<P: Problem> State for BatchedHillClimbing<P> {}
# impl<P: Problem> HasState for BatchedHillClimbing<P> {
#     type State = Self;
#     fn get_state(&self) -> &Self { self }
#     fn get_mut_state(&mut self) -> &mut Self { self }
#     fn from_state(state: Self) -> Self { state }
# }
use tantale::{
    algos::utils::with_rng,
    basic_bounds,
    core::{
        HasInfo, HasX, Sampled, Sampler, Searchspace, SolutionShape, Unit,
        aliases::{Accum, CodElem},
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
impl<P> Sampler<P, Unit> for BatchedHillClimbing<P>
where
    Accum<P>: SingleAccumulator<CompShape<P>>,
    CodElem<P>: ElemSingle + Ord,
{
    type Info = BHCInfo;

    fn sample_info(&mut self, scp: &P::Scp, acc: &Accum<P>)
        -> Option<(Shape<P>, Sampled<BHCInfo>)>
    {
        if let Some(candidate) = self.batch.pop() {
            self.pending += 1;
            Some(candidate.into())
        } else if self.pending > 0 {
            None
        } else {
            self.iterations += 1;
            self.batch = (0..self.batch_size)
                .map(|_| {
                    let shape = match acc.get_best() {
                        Some(best) => {
                            // find_info extracts BHCInfo even if wrapped by other metadata.
                            let _info = best.sinfo().find_info::<BHCInfo>().unwrap();
                            let x = best.get_sopt().ref_x();
                            scp.new_opt(with_rng(|rng| neighbor(x, rng)))
                        }
                        None => with_rng(|rng| scp.sample_pair(rng)),
                    };
                    let info = Sampled::Fresh(BHCInfo::new(self.iterations).into());
                    ShapeWithInfo(shape, info)
                })
                .collect();
            self.sample_info(scp, acc)
        }
    }

    fn update(&mut self, _x: &CompShape<P>, _scp: &P::Scp, _acc: &Accum<P>) {
        self.pending = self.pending.saturating_sub(1);
    }
}
```

Points to note:
- `find_info::<BHCInfo>()` reads the metadata of the best result even when
  another layer wraps it (for example an outer or inner sampler adding its own info).
- The best result is read from the accumulator, which is why
  `Accum<P>: SingleAccumulator<CompShape<P>>` is required.
- The batch size must be positive. With zero, the generated batch is empty and
  the recursive `sample_info` call never terminates in this example.
- The neighborhood needs at least one optimizer coordinate, as in the
  sequential guide.
- Returning `None` means "wait", not "finished". A sampler that returns `None`
  forever with nothing pending can stop the experiment.

## The `Batched` trait

The example also implements `Batched<P, Unit>` (same [`basic_bounds`](crate::macros::basic_bounds) and
`where` clauses as the sampler). The trait exposes the queues to the rest of
the library and requires:

- `batch_size` and `set_batch_size`: read and change the batch size at runtime.
- `batch` and `mut_batch`: the candidates waiting to be served.
- `comp_batch` and `mut_comp_batch`: computed candidates (unused by this
  sampler, but required).

On top of these, the trait provides `add_to_batch`, `pop_from_batch`,
`add_to_comp_batch` and `pop_from_comp_batch`. Each accessor simply returns
the matching field. [`BatchRandomSearch`](crate::algos::BatchRandomSearch) is another implementor, backed by its
`batch_uncomputed` and `batch_computed` fields. The example imports the trait
from `tantale::core::optimizer::opt::Batched`; the sampler logic above works
without it, so implement it when the queue must be reachable through the trait.

```rust
# use std::{fmt::Debug, sync::Arc};
# use rand::{RngExt, rngs::StdRng};
# use serde::{Deserialize, Serialize};
# use tantale::core::{
#     HasState, Problem, SolInfo, State,
#     aliases::{CompShape, Shape},
#     utils::ShapeWithInfo,
# };
# use tantale::macros::{Recordable, Schema};
# #[derive(Schema, Recordable, Serialize, Deserialize, Debug, Default, Clone, Copy)]
# pub struct BHCInfo { pub iterations: usize }
# impl SolInfo for BHCInfo {}
# impl BHCInfo { pub fn new(iterations: usize) -> Self { Self { iterations } } }
# #[derive(Serialize, Deserialize)]
# #[serde(bound = "ShapeWithInfo<Shape<P>, BHCInfo>: Serialize + for<'a> Deserialize<'a>")]
# pub struct BatchedHillClimbing<P: Problem> {
#     iterations: usize,
#     batch_size: usize,
#     batch: Vec<ShapeWithInfo<Shape<P>, BHCInfo>>,
#     comp_batch: Vec<CompShape<P>>,
#     pending: usize,
# }
# impl<P: Problem> Clone for BatchedHillClimbing<P> {
#     fn clone(&self) -> Self {
#         Self { iterations: self.iterations, batch_size: self.batch_size,
#                batch: self.batch.clone(), comp_batch: self.comp_batch.clone(),
#                pending: self.pending }
#     }
# }
# impl<P: Problem> Debug for BatchedHillClimbing<P> {
#     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
#         f.debug_struct("BatchedHillClimbing").finish()
#     }
# }
# impl<P: Problem> State for BatchedHillClimbing<P> {}
# impl<P: Problem> HasState for BatchedHillClimbing<P> {
#     type State = Self;
#     fn get_state(&self) -> &Self { self }
#     fn get_mut_state(&mut self) -> &mut Self { self }
#     fn from_state(state: Self) -> Self { state }
# }
# use tantale::{
#     algos::utils::with_rng,
#     basic_bounds,
#     core::{
#         HasInfo, HasX, Sampled, Sampler, Searchspace, SolutionShape, Unit,
#         aliases::{Accum, CodElem},
#         domain::{ElemSingle, SingleAccumulator},
#     },
# };
# 
# fn neighbor(x: &[f64], rng: &mut StdRng) -> Arc<[f64]> {
#     let mut next = x.to_vec();
#     let i = rng.random_range(0..next.len());
#     next[i] = (next[i] + rng.random_range(-0.1..0.1)).clamp(0.0, 1.0);
#     Arc::from(next)
# }
# 
# #[basic_bounds(P, Unit)]
# impl<P> Sampler<P, Unit> for BatchedHillClimbing<P>
# where
#     Accum<P>: SingleAccumulator<CompShape<P>>,
#     CodElem<P>: ElemSingle + Ord,
# {
#     type Info = BHCInfo;
# 
#     fn sample_info(&mut self, scp: &P::Scp, acc: &Accum<P>)
#         -> Option<(Shape<P>, Sampled<BHCInfo>)>
#     {
#         if let Some(candidate) = self.batch.pop() {
#             self.pending += 1;
#             Some(candidate.into())
#         } else if self.pending > 0 {
#             None
#         } else {
#             self.iterations += 1;
#             self.batch = (0..self.batch_size)
#                 .map(|_| {
#                     let shape = match acc.get_best() {
#                         Some(best) => {
#                             // find_info extracts BHCInfo even if wrapped by other metadata.
#                             let _info = best.sinfo().find_info::<BHCInfo>().unwrap();
#                             let x = best.get_sopt().ref_x();
#                             scp.new_opt(with_rng(|rng| neighbor(x, rng)))
#                         }
#                         None => with_rng(|rng| scp.sample_pair(rng)),
#                     };
#                     let info = Sampled::Fresh(BHCInfo::new(self.iterations).into());
#                     ShapeWithInfo(shape, info)
#                 })
#                 .collect();
#             self.sample_info(scp, acc)
#         }
#     }
#
#     fn update(&mut self, _x: &CompShape<P>, _scp: &P::Scp, _acc: &Accum<P>) {
#         self.pending = self.pending.saturating_sub(1);
#     }
# }

use tantale::core::optimizer::Batched;

#[basic_bounds(P, Unit)]
impl<P> Batched<P, Unit> for BatchedHillClimbing<P>
where
    Accum<P>: SingleAccumulator<CompShape<P>>,
    CodElem<P>: ElemSingle + Ord,
{
    fn batch_size(&self) -> usize { self.batch_size }
    fn set_batch_size(&mut self, size: usize) { self.batch_size = size; }
    fn batch(&self) -> &[ShapeWithInfo<Shape<P>, BHCInfo>] { self.batch.as_slice() }
    fn mut_batch(&mut self) -> &mut Vec<ShapeWithInfo<Shape<P>, BHCInfo>> { &mut self.batch }
    fn comp_batch(&self) -> &[CompShape<P>] { self.comp_batch.as_slice() }
    fn mut_comp_batch(&mut self) -> &mut Vec<CompShape<P>> { &mut self.comp_batch }
}
```

This block is reviewed against the example only; the example itself compiles
(`cargo check -p tantale_examples --examples`).

## Use in an experiment

Supply a problem whose optimizer domain is `Unit`, instantiate
`BatchedHillClimbing::new(batch_size)` and pass both to
`Experiment::new(problem, sampler, stop, recorder, checkpointer)`, or to
[`ParExperiment`](crate::core::ParExperiment) for threaded or MPI runs, where several candidates of a batch
are evaluated at the same time and idle workers wait on `None`. The problem's
solution metadata type must match `BHCInfo`; type inference normally
determines it from the sampler. The source example has an empty `main`; it
does not itself run an experiment. The two Rust blocks that define the
sampler are compiled as RustDoc tests with `cargo test -p tantale --doc`.
