//! The [Hyperband](https://arxiv.org/pdf/1603.06560) algorithm for multi-fidelity hyperparameter optimization.
//!
//! # Overview
//! 
//! The objective of Hyperband is to efficiently allocate a fixed budget of resources 
//! (e.g., time, epochs, etc.) across a set of hyperparameter configurations to identify the best performing configuration.
//! Hyperband achieves this by iteratively evaluating a large number of configurations with a small budget 
//! and then successively halving the number of configurations while increasing the budget for the remaining ones.
//!
//! # Pseudo-code
//! 
//! **Asynchronous Successive Halving (ASHA)**
//! ---
//! **Inputs**
//! 1. &emsp; $\mathcal{X}$ &emsp;&emsp; *A [searchspace](tantale_core::Searchspace)*
//! 2. &emsp; $b_0$ &emsp;&emsp; *Initial budget*
//! 3. &emsp; $b_{\text{max}}$ &emsp;&emsp; *Maximum budget*
//! 4. &emsp; $\eta$ &emsp;&emsp; *Scaling*
//! 5. &emsp;
//! 6. &emsp; $B = [b_0, b_0\cdot\eta^1,b_0\cdot\eta^2, \cdots, b_{\text{max}}]$ &emsp; *Precompute the budget levels*
//! 7. &emsp; $\mathcal{R} = (R_i)_{i \in [0,\cdots,|B|]}\enspace \text{s.t. } R_i = \emptyset$ &emsp; *Initialize empty rungs for each budget level*
//! 8. &emsp;
//! 9. &emsp; **function** worker() &emsp; *Each worker runs this function asynchronously*
//! 10. &emsp; &emsp; **while** not stop **do**
//! 11. &emsp; &emsp;&emsp; $(x,i) \gets \text{generate()}$ &emsp; *Generate a new [solution](tantale_core::Solution) to evaluate at the budget level $B_i$*
//! 12. &emsp; &emsp;&emsp; $y \gets f(x;B_i)$ &emsp; *Evaluate $x$ with [fidelity](tantale_core::Fidelity) $B_i$*
//! 13. &emsp; &emsp;&emsp; $R_i \gets R_i \cup \{(x,y)\}$ &emsp; *Add the generated solution to the rung $R_i$*
//! 14. &emsp;
//! 15. &emsp; **function** generate() &emsp; *Generates a new [solution](tantale_core::Solution) to evaluate at the appropriate budget level*
//! 16. &emsp; &emsp; $i \gets \lvert B \rvert - 1$ &emsp; *Start from the highest budget level*
//! 17. &emsp; &emsp; $\mathbf{x} \gets \emptyset$ &emsp; *Initialize empty set for top $k$ solutions*
//! 18. &emsp; &emsp; **while** $\lvert \mathbf{x} \rvert = 0$ **and** $i > 0$ **do**
//! 19. &emsp; &emsp; &emsp; $\mathbf{x} \gets \text{Top}_k\left(R_i,\left\lfloor \frac{\lvert R_i \rvert }{\eta} \right\rfloor\right)$ *Select the top $\left\lfloor \frac{\lvert R_i \rvert }{\eta} \right\rfloor$ best [computed](tantale_core::Computed)*
//! 20. &emsp; &emsp; &emsp; $i \gets i - 1$ &emsp; *Move to the next lower budget level*
//! 21. &emsp; &emsp; **if** $i = 0$ **then**
//! 22. &emsp; &emsp;&emsp; **return** $(\text{Random}(\mathcal{X},1),0)$
//! 23. &emsp; &emsp; **else**
//! 24. &emsp; &emsp;&emsp; $R_i \gets R_i \setminus \mathbf{x}_0$ &emsp; *Remove the selected solutions from the rung*
//! 25. &emsp; &emsp;&emsp; **return** $(\mathbf{x}_0, i)$
//! ---
//!
//! # Type Parameters
//!
//! The algorithm is generic over:
//! - Output types satisfying [`FidOutcome`](tantale_core::FidOutcome) for multi-fidelity support
//! - [`Searchspace`](tantale_core::Searchspace) over randomly samplable [`Domain`](tantale_core::Domain) generating candidates with [`HasStep`] and [`HasFidelity`] traits
//!
//! # Example
//!
//! ```ignore
//! let sh = ASHA::new(
//!     budget_min,      // Minimum resource level (e.g., epochs=1)
//!     budget_max,      // Maximum resource level (e.g., epochs=100)
//!     scaling,  // Reduction factor (e.g., 2.0 or 3.0)
//! );
//! ```
//!
//! # Note
//!
//! In our case, Successive Halving does not stop when the final rung is evaluated. If so, then it generates a new initial batch and starts a new run.
//! This allows compatibility with the [`Stop`](tantale_core::Stop) criterion, which can be used to stop the optimization after a certain number of iteration, evaluations, time...
//!
//! # References
//!
//! Successive Halving is based on the work of [Li et al. (2018)](https://arxiv.org/pdf/1810.05934).

use std::marker::PhantomData;

use tantale_core::{
    Codomain, Criteria, EmptyInfo, FidOutcome, Fidelity, FidelitySol, FuncState, HasFidelity, HasStep, IntoComputed, LinkOpt, OptState, Optimizer, RawObj, SId, Searchspace, SequentialOptimizer, SingleCodomain, SolutionShape, Step, Stepped, searchspace::OptionCompShape, solution::Uncomputed
};
use serde::ser::{Serialize, Serializer, SerializeStruct};

/// Creates a codomain for Successive Halving optimization.
///
/// Constructs a [`SingleCodomain`](tantale_core::SingleCodomain) from a single-objective
/// [`Criteria`](tantale_core::Criteria).
///
/// # Arguments
///
/// * `extractor` - A [`Criteria`](tantale_core::Criteria) defining how to extract the
///   optimization objective from the [`Outcome`](tantale_core::Outcome).
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

/// Internal state of the [`Asha`] optimizer.
///
/// This structure maintains all essential information needed to resume an optimization
/// across checkpoints. It encodes the core parameters of the algorithm and current iteration.
pub struct HyperbandState<Optim,Out,Scp>
where
    Optim : Optimizer<FidelitySol<SId,Scp::Opt,EmptyInfo>,SId,Scp::Opt,Out,Scp, SInfo = EmptyInfo>,
    Out : FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>
{
    pub budget_min: f64,
    pub budget_max: f64,
    pub scaling: f64,
    pub inner: Optim,
    _out: PhantomData<Out>,
    _scp: PhantomData<Scp>,
}

impl<Optim,Out,Scp>  Serialize for HyperbandState<Optim,Out,Scp> 
where
    Optim : Optimizer<FidelitySol<SId,Scp::Opt,EmptyInfo>,SId,Scp::Opt,Out,Scp, SInfo = EmptyInfo> + Serialize,
    Out : FidOutcome + Serialize,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo> + Serialize
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("HyperbandState", 4)?;
        state.serialize_field("budget_min", &self.budget_min)?;
        state.serialize_field("budget_max", &self.budget_max)?;
        state.serialize_field("scaling", &self.scaling)?;
        state.serialize_field("inner", &self.inner.get_state())?;
        state.end()
    }
}

impl<Optim,Out,Scp> OptState for HyperbandState<Optim,Out,Scp> 
where
    Optim : Optimizer<FidelitySol<SId,Scp::Opt,EmptyInfo>,SId,Scp::Opt,Out,Scp, SInfo = EmptyInfo>,
    Out : FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>
{}

pub struct Hyperband<Optim,Out,Scp>(pub HyperbandState<Optim, Out, Scp>)
where
    Optim : Optimizer<FidelitySol<SId,Scp::Opt,EmptyInfo>,SId,Scp::Opt,Out,Scp, SInfo = EmptyInfo>,
    Out : FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>;

impl<Optim,Out,Scp> Hyperband<Optim,Out,Scp>
where
    Optim : Optimizer<FidelitySol<SId,Scp::Opt,EmptyInfo>,SId,Scp::Opt,Out,Scp, SInfo = EmptyInfo>,
    Out : FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    pub fn new(budget_min: f64, budget_max: f64, scaling: f64, opt: Optim) -> Self {
        let state = HyperbandState {
            budget_min,
            budget_max,
            scaling,
            opt_state: opt.get_mut_state().clone(),
        };
        Self(state, opt)
    }
}