# Creating a new Optimizer

We consider the [Asynchronous Successive Halving](https://arxiv.org/abs/1502.07943) algorithm described by the following pseudo-code:

**Asynchronous Successive Halving (ASHA)**
---
**Inputs**
1. &emsp; $\mathcal{X}$ &emsp;&emsp; *A [searchspace](tantale_core::Searchspace)*
2. &emsp; $b_0$ &emsp;&emsp; *Initial budget*
3. &emsp; $b_{\text{max}}$ &emsp;&emsp; *Maximum budget*
4. &emsp; $\eta$ &emsp;&emsp; *Scaling*
5. &emsp;
6. &emsp; $B = [b_0, b_0\cdot\eta^1,b_0\cdot\eta^2, \cdots, b_{\text{max}}]$ &emsp; *Precompute the budget levels*
7. &emsp; $\mathcal{R} = (R_i)_{i \in [0,\cdots,|B|]}\enspace \text{s.t. } R_i = \emptyset$ &emsp; *Initialize empty rungs for each budget level*
8. &emsp;
9. &emsp; **function** worker() &emsp; *Each worker runs this function asynchronously*
10. &emsp; &emsp; **while** not stop **do**
11. &emsp; &emsp;&emsp; $(x,i) \gets \text{generate()}$ &emsp; *Generate a new [solution](tantale_core::Solution) to evaluate at the budget level $B_i$*
12. &emsp; &emsp;&emsp; $y \gets f(x;B_i)$ &emsp; *Evaluate $x$ with [fidelity](tantale_core::Fidelity) $B_i$*
13. &emsp; &emsp;&emsp; $R_i \gets R_i \cup \{(x,y)\}$ &emsp; *Add the generated solution to the rung $R_i$*
14. &emsp;
15. &emsp; **function** generate() &emsp; *Generates a new [solution](tantale_core::Solution) to evaluate at the appropriate budget level*
16. &emsp; &emsp; $i \gets \lvert B \rvert - 1$ &emsp; *Start from the highest budget level*
17. &emsp; &emsp; $\mathbf{x} \gets \emptyset$ &emsp; *Initialize empty set for top $k$ solutions*
18. &emsp; &emsp; **while** $\lvert \mathbf{x} \rvert = 0$ **and** $i > 0$ **do**
19. &emsp; &emsp; &emsp; $\mathbf{x} \gets \text{Top}_k\left(R_i,\left\lfloor \frac{\lvert R_i \rvert }{\eta} \right\rfloor\right)$ *Select the top $\left\lfloor \frac{\lvert R_i \rvert }{\eta} \right\rfloor$ best [computed](tantale_core::Computed)*
20. &emsp; &emsp; &emsp; $i \gets i - 1$ &emsp; *Move to the next lower budget level*
21. &emsp; &emsp; **if** $i = 0$ **then**
22. &emsp; &emsp;&emsp; **return** $(\text{Random}(\mathcal{X},1),0)$
23. &emsp; &emsp; **else**
24. &emsp; &emsp;&emsp; $R_i \gets R_i \setminus \mathbf{x}_0$ &emsp; *Remove the selected solutions from the rung*
25. &emsp; &emsp;&emsp; **return** $(\mathbf{x}_0, i)$
---

## Preliminary technical details

The optimizer uses a thread-local [`StdRng`] for random sampling. The RNG is not part of the optimizer state, as it cannot be serialized or deserialized. The [`StdRng`] is defined at the module level as follows:
```rust
use rand::{SeedableRng, rngs::StdRng};

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_os_rng());
}
```
It is called with a private method `with_rng` that takes a closure, allowing the optimizer to perform random sampling while keeping the RNG separate from the optimizer state:
```rust,ignore
self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into()))
```

## Defining the state

First, we need to define the [`OptState`](crate::core::OptState). This marker trait 
defines all necessary information for each iteration of the optimizer.
It is also used for checkpointing, resuming an experiment, and creating the optimizer object from its state.

According to the previous pseudo-code ASHA requires:
- A [`Searchspace`](crate::core::Optimizer): It is given as a parameter to the [`Optimizer`](crate::core::Optimizer) at every iteration. So, there is no need to specify this into [`OptState`](crate::core::OptState).
- $B$, all available budgets: this must specified within [`OptState`](crate::core::OptState).
- $\eta$, the scaling factor: Used for initialization and to compute the number of minimum candidate with a rung.
- $\mathcal{R}$, the different rungs made of [`Computed`](crate::core::Computed)s.

AshaState is generic over all [`SolutionShape`](crate::core::SolutionShape), i.e. pairs of `Obj` and `Opt` [`Solution`](crate::core::Solution).
But, these are constrained by [`HasStep`](crate::core::HasStep) and [`HasFidelity`](crate::core::HasFidelity) because we are within the multi-fidelity optimization framework. To simplify things, it must also implements [`Ord`](std::cmp::Ord) (based on [`HasY`](crate::core::HasY) of [`Computed`](crate::core::Computed)) to simplify the Top k selection of the best [`Computed`](crate::core::Computed).

A [`SolutionShape`](crate::core::SolutionShape) also implements `Serializable` and `Deserializable`, which explains this:
```rust,ignore
#[serde(bound(serialize = "SShape: Serialize", deserialize = "SShape: for<'a> Deserialize<'a>"))]
```

Then, the [`OptState`](crate::core::OptState) is given by:

```rust
use tantale::core::{SolutionShape, SId, EmptyInfo, HasStep, HasFidelity, HasY};
use tantale::macros::OptState;
use std::cmp::Ord;
use serde::{Serialize,Deserialize};

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SShape: Serialize",
    deserialize = "SShape: for<'a> Deserialize<'a>",
))]
pub struct AshaState<SShape>
where
    SShape: SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + HasY + Ord,
{
    pub budgets: Vec<f64>,
    pub scaling: f64,
    pub rungs: Vec<Vec<SShape>>,
}
```

## Defining per-iteration metadata

ASHA does not have per-iteration metadata.

## Defining per-solution metadata

ASHA does not have per-solution metadata, associated fidelities/budgets are stored within a [`FidelitySol`](crate::core::FidelitySol)

## Characterize the optimizer

The next step is to define the [`Optimizer`](crate::core::Optimizer). This trait does not yet define the iteration itself, but the different types characterizing the optimizer.
This include:
* PSol: A generic defining what kind of solution the optimizer handles. For instance, [`BaseSol`](crate::core::BaseSol), or [`FidelitySol`](crate::core::FidelitySol).
* SolId: A generic [`Id`](crate::core::Id) type, to make a solution [`Solution`](crate::core::Solution) unique.
* Opt: A generic [`Domain`](crate::core::Domain) to constrain or not an optimizer's search domain type (e.g. continus, natural...).
* Out: Should always be generic, it corresponds to the user-defined [`Outcome`](crate::core::Outcome) of the function to optimize.
* Scp: A generic over a [`Searchspace`](crate::core::Searchspace). It can be specified if the optimizer works for some kind of searchspaces.

Then, multiple associated types have to be defined:
* [`State`](crate::core::Optimizer::State): The [`OptState`](crate::core::OptState) described earlier
* [`Cod`](crate::core::Optimizer::Cod): The [`Codomain`](crate::core::Codomain) the algorithm is optimizing (e.g. single or multi-objectives)
* [`SInfo`](crate::core::SolInfo): Some meta-data associated to each unique solution
* [`Info`](crate::core::OptInfo): Per-iteration meta-data

Finally, two methods have to be written:
* `get_state`: returns the current [`OptState`](crate::core::OptState) of the optimizer.
* `from_state`: creates an instance of the optimizer from the [`OptState`](crate::core::OptState).

### Creating Successive Halving struct

ASHA only requires its [`OptState`], i.e. [`AshaState`].

Then, we can implement some methods for a better user usage. For example, a `new` builder method. Or, a `codomain` method
to help creating the right [`Codomain`](crate::core::Codomain) for the optimizer.

``` rust
# use tantale::core::{SolutionShape, SId, EmptyInfo, HasStep, HasFidelity, HasY};
# use tantale::macros::OptState;
# use std::cmp::Ord;
# use serde::{Serialize,Deserialize};
# use rand::{SeedableRng, rngs::StdRng};
# 
# thread_local! {
#     static THREAD_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_os_rng());
# }
#
# #[derive(Serialize, Deserialize)]
# #[serde(bound(
#     serialize = "SShape: Serialize",
#     deserialize = "SShape: for<'a> Deserialize<'a>",
# ))]
# pub struct AshaState<SShape>
# where
#     SShape: SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + HasY + Ord,
# {
#     pub budgets: Vec<f64>,
#     pub scaling: f64,
#     pub rungs: Vec<Vec<SShape>>,
# }

use tantale::core::{Codomain, Criteria, FidOutcome, SingleCodomain};

pub fn codomain<Cod, Out>(extractor: Criteria<Out>) -> Cod
where
    Cod: Codomain<Out> + From<SingleCodomain<Out>>,
    Out: FidOutcome,
{
    let out = SingleCodomain {
        y_criteria: extractor,
    };
    out.into()
}

pub struct Asha(pub AshaState);

impl AshaState {
    pub fn new(budget_min: f64, budget_max: f64, scaling: f64) -> Self {
        assert!(scaling >= 1.0, "Scaling factor must be >= 1.0");
        assert!(budget_min > 0.0, "Minimum budget must be > 0.0");
        assert!(budget_max > budget_min,"Maximum budget must be > minimum budget");
        
        // Create all different budgets
        let mut budgets: Vec<f64> = (0..)
            .map(|i| budget_min * scaling.powi(i))
            .take_while(|&b| b < budget_max)
            .collect();
        //If final budget does not round to budget_max, add budget_max as final budget level
        if budgets.last().unwrap().round() != budget_max {
            budgets.push(budget_max);
        } else {
            // else rounds final budget to budget_max, round to budget_max
            budgets.last_mut().unwrap().round();
        }
        
        let length = budgets.len();
        Asha(
            AshaState {
                budgets,
                scaling,
                // Create i rungs, even if rung 0 will not be used, it simplifies computations
                rung: (0..length).map(|_| Vec::new()).collect(),
            },
        )
    }
}
```


### Implementing Optimizer trait

We consider the `AshaState` previously described. Asha only requires the `Opt` [`Domain`](crate::core::Domain) to
be samplable. Therefore, the [`Optimizer`](crate::core::Optimizer) can be generic over this domain.
This is modeled by `Scp::Opt` equal to the type alias `LinkOpt<Scp>`. It means that ASHA can be used whatever
the `Opt` [`Domain`](crate::core::Domain) is. ASHA is a multi-fidelity optimizer, working with [`FidelitySol`](crate::core::FidelitySol) solution type. It is also generic over any [`FidOutcome`](crate::core::FidOutcome), i.e. any [`Outcome`](crate::core::Outcome) containing a [`Step`](crate::core::Step).

Notice the bound on `<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>`. It specifies that the computed version of a [`SolShape`](crate::core::SolShape) must follows the same bound described within `AshaState`.

```rust
# use tantale::core::{SolutionShape, SId, EmptyInfo, HasStep, HasFidelity, HasY};
# use tantale::macros::OptState;
# use std::cmp::Ord;
# use serde::{Serialize,Deserialize};
# use rand::{SeedableRng, rngs::StdRng};
# 
# thread_local! {
#     static THREAD_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_os_rng());
# }
#
# #[derive(Serialize, Deserialize)]
# #[serde(bound(
#     serialize = "SShape: Serialize",
#     deserialize = "SShape: for<'a> Deserialize<'a>",
# ))]
# pub struct AshaState<SShape>
# where
#     SShape: SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + HasY + Ord,
# {
#     pub budgets: Vec<f64>,
#     pub scaling: f64,
#     pub rungs: Vec<Vec<SShape>>,
# }
# 
# use tantale::core::{Codomain, Criteria, FidOutcome, SingleCodomain};
# 
# pub fn codomain<Cod, Out>(extractor: Criteria<Out>) -> Cod
# where
#     Cod: Codomain<Out> + From<SingleCodomain<Out>>,
#     Out: FidOutcome,
# {
#     let out = SingleCodomain {
#         y_criteria: extractor,
#     };
#     out.into()
# }
# 
# pub struct Asha(pub AshaState);
# 
# impl AshaState {
#     pub fn new(budget_min: f64, budget_max: f64, scaling: f64) -> Self {
#         assert!(scaling >= 1.0, "Scaling factor must be >= 1.0");
#         assert!(budget_min > 0.0, "Minimum budget must be > 0.0");
#         assert!(budget_max > budget_min,"Maximum budget must be > minimum budget");
#         
#         // Create all different budgets
#         let mut budgets: Vec<f64> = (0..)
#             .map(|i| budget_min * scaling.powi(i))
#             .take_while(|&b| b < budget_max)
#             .collect();
#         //If final budget does not round to budget_max, add budget_max as final budget level
#         if budgets.last().unwrap().round() != budget_max {
#             budgets.push(budget_max);
#         } else {
#             // else rounds final budget to budget_max, round to budget_max
#             let last = budgets.last_mut().unwrap();
#             *last = last.round();
#         }
#         
#         let length = budgets.len();
#         Asha(
#             AshaState {
#                 budgets,
#                 scaling,
#                 // Create i rungs, even if rung 0 will not be used, it simplifies computations
#                 rung: (0..length).map(|_| Vec::new()).collect(),
#             },
#         )
#     }
# }

use tantale::core::{EmptyInfo, FidelitySol, Optimizer, SId, Searchspace, LinkOpt};

impl<Out, Scp> Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for Asha<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>:
        SolutionShape<SId,EmptyInfo> + HasStep + HasFidelity + Ord,
{
    type State = AshaState<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo; // No metadata
    type Info = EmptyInfo; // No metadata

    fn get_state(&self) -> &Self::State {
        &self.0
    }
    
    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Asha(state)
    }
}
```

### Implementing SequentialOptimizer trait

We have implemented the Optimizer trait to characterize ASHA.
Now we can define computations of an iteration.
ASHA generates on demand [`FidelitySol`](crate::core::FidelitySol).So, ASHA is a [`SequentialOptimizer`](crate::core::SequentialOptimizer).
We have to define one functions:
- `step`: the usual iteration of the algorithm after initialization. It should be able ot generate solutions when it receives one or no
  [`Computed`](crate::core::Computed).

```rust
# use tantale::core::{SolutionShape, SId, EmptyInfo, HasStep, HasFidelity, HasY};
# use tantale::macros::OptState;
# use std::cmp::Ord;
# use serde::{Serialize,Deserialize};
# use rand::{SeedableRng, rngs::StdRng};
# 
# thread_local! {
#     static THREAD_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_os_rng());
# }
#
# #[derive(Serialize, Deserialize)]
# #[serde(bound(
#     serialize = "SShape: Serialize",
#     deserialize = "SShape: for<'a> Deserialize<'a>",
# ))]
# pub struct AshaState<SShape>
# where
#     SShape: SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + HasY + Ord,
# {
#     pub budgets: Vec<f64>,
#     pub scaling: f64,
#     pub rungs: Vec<Vec<SShape>>,
# }
# 
# use tantale::core::{Codomain, Criteria, FidOutcome, SingleCodomain};
# 
# pub fn codomain<Cod, Out>(extractor: Criteria<Out>) -> Cod
# where
#     Cod: Codomain<Out> + From<SingleCodomain<Out>>,
#     Out: FidOutcome,
# {
#     let out = SingleCodomain {
#         y_criteria: extractor,
#     };
#     out.into()
# }
# 
# pub struct Asha(pub AshaState);
# 
# impl AshaState {
#     pub fn new(budget_min: f64, budget_max: f64, scaling: f64) -> Self {
#         assert!(scaling >= 1.0, "Scaling factor must be >= 1.0");
#         assert!(budget_min > 0.0, "Minimum budget must be > 0.0");
#         assert!(budget_max > budget_min,"Maximum budget must be > minimum budget");
#         
#         // Create all different budgets
#         let mut budgets: Vec<f64> = (0..)
#             .map(|i| budget_min * scaling.powi(i))
#             .take_while(|&b| b < budget_max)
#             .collect();
#         budgets.push(budget_max); // Add the last maximum budget
#         
#         let length = budgets.len();
#         Asha(
#             AshaState {
#                 budgets,
#                 scaling,
#                 // Create i rungs, even if rung 0 will not be used, it simplifies computations
#                 rung: (0..length).map(|_| Vec::new()).collect(),
#             },
#         )
#     }
# }
# 
# impl<Out, Scp> Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
#     for Asha<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>
# where
#     Out: FidOutcome,
#     Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
#     Scp::SolShape: HasStep + HasFidelity,
#     <Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>:
#         SolutionShape<SId,EmptyInfo> + HasStep + HasFidelity + Ord,
# {
#     type State = AshaState<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>;
#     type Cod = SingleCodomain<Out>;
#     type SInfo = EmptyInfo; // No metadata
#     type Info = EmptyInfo; // No metadata
# 
#     fn get_state(&mut self) -> &Self::State {
#         &self.0
#     }
# 
#     fn from_state(state: Self::State) -> Self {
#         Asha(state)
#     }
# }

use tantale_core::{SequentialOptimizer, Codomain, OptionCompShape, FuncState, HasFidelity, HasStep, IntoComputed, RawObj, Step, Stepped};


impl<Out, Scp, FnState>
    SequentialOptimizer<
        FidelitySol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
    > for Asha<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>:
        SolutionShape<SId,EmptyInfo> + HasStep + HasFidelity +Ord,
    FnState: FuncState,
{
    fn step(
        &mut self,
        x: OptionCompShape<Scp, FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Self::SInfo, Self::Cod, Out>,
        scp: &Scp,
    ) -> Scp::SolShape {
        // If input is not empty (a solution has been computed)
        if let Some(comp) = x
        {
            // If this solution is partially computed, then store it within the next rung.
            if let Step::Partially(_) = comp.step() {
                // The idx of the budget cannot be stored within the solution
                let idx = self.0.budgets.iter().position(|&b| b == comp.fidelity().0).unwrap();
                self.0.rung[idx + 1].push(comp); // Store it within the next rung
            }

            let mut i = self.0.budgets.len() - 1;
            // Compute top k for final rung. (should be 0 if we are within the first iterations)
            let mut k = (self.0.rung[i].len() as f64 / self.0.scaling) as usize;
            // Find a rung with promotable solutions
            while k == 0 && i > 0 {
                i -= 1;
                k = (self.0.rung[i].len() as f64 / self.0.scaling) as usize;
            }
            // If no rung was found, generate a random solution
            if k == 0 {
                let mut p = self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into()));
                p.set_fidelity(self.0.budgets[0]); // Modify default fidelity to minimum budget
                p // return p
            } else {
                // Select the top k (modify in place rung[i]), last elements are the top k
                self.0.rung[i].select_nth_unstable(k);
                // Pop the last element /!\ the rung is not sorted by select_nth_unstable, only partitioned
                let (mut p,_): (Scp::SolShape, _) = IntoComputed::extract(self.0.rung[i].pop().unwrap()); 
                p.set_fidelity(self.0.budgets[i]); // Modify previous fidelity with new budget
                p
            }
        } else {// If input is None (no computed, e.g. initialization of ASHA)
            // Randomly sample a new candidate with minimum budget
            let mut p = self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into()));
            p.set_fidelity(self.0.budgets[0]);
            p
        }
    }   
}

```