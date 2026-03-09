use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, cmp::Ord};
use tantale::core::{EmptyInfo, HasFidelity, HasStep, SId, SolutionShape};
use tantale::macros::OptState;

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_os_rng());
}
#[derive(OptState, Serialize, Deserialize)]
#[serde(bound(
    serialize = "SShape: Serialize",
    deserialize = "SShape: for<'a> Deserialize<'a>",
))]
pub struct AshaState<SShape>
where
    SShape: SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + Ord,
{
    // A vector of budget levels corresponding to the halving rounds.
    pub budgets: Vec<f64>,
    // Scaling factor ($\eta$) by which the budget is multiplied at each stage.
    pub scaling: f64,
    /// A vector of vectors representing the rungs of the Successive Halving process.
    pub rung: Vec<Vec<SShape>>,
    // The current budget level index being processed. This is used to track which rung is currently active for promotions and evaluations.
    pub current_budget: f64,
}

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

pub struct Asha<SShape>(pub AshaState<SShape>)
where
    SShape: SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + Ord;

impl<SShape> Asha<SShape>
where
    SShape: SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + Ord,
{
    pub fn new(budget_min: f64, budget_max: f64, scaling: f64) -> Self {
        assert!(scaling >= 1.0, "Scaling factor must be >= 1.0");
        assert!(budget_min > 0.0, "Minimum budget must be > 0.0");
        assert!(
            budget_max > budget_min,
            "Maximum budget must be > minimum budget"
        );
        let mut budgets: Vec<f64> = (0..)
            .map(|i| budget_min * scaling.powi(i))
            .take_while(|&b| b <= budget_max)
            .collect();
        //If final budget is not budget_max, modify final budget to be budget_max
        if *budgets.last().unwrap() != budget_max {
            let last = budgets.last_mut().unwrap();
            *last = budget_max;
        }

        let length = budgets.len();
        let current_budget = budgets[0];
        Asha(AshaState {
            budgets,
            scaling,
            rung: (0..length).map(|_| Vec::new()).collect(),
            current_budget,
        })
    }
    pub fn with_rng<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut StdRng) -> T,
    {
        THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
    }
}

use tantale::core::{FidelitySol, IntoComputed, LinkOpt, Optimizer, Searchspace};

impl<Out, Scp> Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for Asha<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>:
        SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + Ord,
{
    type State = AshaState<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo; // No metadata

    fn get_state(&self) -> &Self::State {
        &self.0
    }

    fn get_mut_state(&mut self) -> &Self::State {
        &mut self.0
    }

    fn from_state(state: Self::State) -> Self {
        Asha(state)
    }
}

use tantale::core::{
    CompAcc, FuncState, OptionCompShape, RawObj, SequentialOptimizer, Step, Stepped,
};

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
        SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + Ord,
    FnState: FuncState,
{
    fn step(
        &mut self,
        x: OptionCompShape<
            Scp,
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
        scp: &Scp,
        _acc: &CompAcc<
            Scp,
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
    ) -> Scp::SolShape {
        // If input is not empty (a solution has been computed)
        if let Some(comp) = x {
            // If this solution is partially computed, then store it within the next rung.
            if let Step::Partially(_) = comp.step() {
                // The idx of the budget cannot be stored within the solution
                let idx = self
                    .0
                    .budgets
                    .iter()
                    .position(|&b| b == comp.fidelity().0)
                    .unwrap();
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
                let (mut p, _): (Scp::SolShape, _) =
                    IntoComputed::extract(self.0.rung[i].pop().unwrap());
                p.set_fidelity(self.0.budgets[i]); // Modify previous fidelity with new budget
                p
            }
        } else {
            // If input is None (no computed, e.g. initialization of ASHA)
            // Randomly sample a new candidate with minimum budget
            let mut p = self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into()));
            p.set_fidelity(self.0.budgets[0]);
            p
        }
    }
}

fn main() {}
