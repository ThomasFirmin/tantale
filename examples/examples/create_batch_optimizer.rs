use serde::{Deserialize, Serialize};
use tantale::macros::OptState;

#[derive(OptState, Serialize, Deserialize)]
pub struct ShaState {
    pub batch: usize,     // batch size
    pub budget_min: f64,  // b0
    pub budget_max: f64,  // bmax
    pub budget: f64,      // b
    pub scaling: f64,     // eta
    pub iteration: usize, // i
}
use tantale::macros::{CSVWritable, OptInfo};

#[derive(OptInfo, CSVWritable, Default, Debug, Serialize, Deserialize)]
pub struct ShaInfo {
    pub iteration: usize,
}

use rand::prelude::ThreadRng;
use tantale::core::{Codomain, Criteria, FidOutcome, SingleCodomain, StepSId};

pub struct Sha(pub ShaState, ThreadRng);

impl Sha {
    pub fn new(batch: usize, budget_min: f64, budget_max: f64, scaling: f64) -> Self {
        assert!(scaling >= 1.0, "Scaling factor must be >= 1.0");
        assert!(budget_min > 0.0, "Minimum budget must be > 0.0");
        assert!(
            budget_max > budget_min,
            "Maximum budget must be > minimum budget"
        );

        let i_max = scaling.powf((budget_max / budget_min).log(scaling));
        assert!(
            batch as f64 >= i_max,
            "Batch size should be greater or equal than {i_max}"
        );

        Sha(
            ShaState {
                batch,
                budget_min,
                budget_max,
                budget: budget_min,
                scaling,
                iteration: 0,
            },
            rand::rng(),
        )
    }

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
}

use tantale::core::{EmptyInfo, FidelitySol, LinkOpt, Optimizer, Searchspace};

impl<Out, Scp> Optimizer<FidelitySol<StepSId, Scp::Opt, EmptyInfo>, StepSId, Scp::Opt, Out, Scp> for Sha
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, EmptyInfo>, StepSId, EmptyInfo>,
{
    type State = ShaState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;

    fn get_state(&self) -> &Self::State {
        &self.0
    }

    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }
    fn from_state(state: Self::State) -> Self {
        Sha(state, rand::rng())
    }
}

use tantale::core::{
    Batch, BatchOptimizer, CompAcc, CompBatch, FuncState, HasFidelity, HasStep, IntoComputed,
    RawObj, SolutionShape, Step, Stepped,
};

impl<Out, Scp, FnState>
    BatchOptimizer<
        FidelitySol<StepSId, Scp::Opt, EmptyInfo>,
        StepSId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, StepSId, EmptyInfo>, Out, FnState>,
    > for Sha
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, EmptyInfo>, StepSId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<StepSId, Self::SInfo> + HasStep + HasFidelity + Ord,
    FnState: FuncState,
{
    type Info = ShaInfo;

    fn first_step(&mut self, scp: &Scp) -> Batch<StepSId, Self::SInfo, Self::Info, Scp::SolShape> {
        let info = ShaInfo {
            iteration: self.0.iteration,
        };
        let pairs: Vec<_> = scp.vec_apply_pair(
            // Use vec_apply_pair to set minimum fidelity
            |mut pair| {
                pair.set_fidelity(self.0.budget);
                pair
            },
            &mut self.1,
            self.0.batch,
            EmptyInfo.into(), // EmptyInfo because to solution metadata
        );
        Batch::new(pairs, info.into())
    }

    fn step(
        &mut self,
        x: CompBatch<
            StepSId,
            Self::SInfo,
            Self::Info,
            Scp,
            FidelitySol<StepSId, Scp::Opt, EmptyInfo>,
            Self::Cod,
            Out,
        >,
        scp: &Scp,
        _acc: &CompAcc<
            Scp,
            FidelitySol<StepSId, Scp::Opt, EmptyInfo>,
            StepSId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
    ) -> Batch<StepSId, Self::SInfo, Self::Info, Scp::SolShape> {
        let (pairs, _) = x.extract(); // Extract component of CompBatch
        // Keep only Partially computed solution
        let mut pairs: Vec<_> = pairs
            .into_iter()
            .filter_map(|comp| match comp.step() {
                Step::Partially(_) => Some(comp),
                _ => None,
            })
            .collect();

        // If no solution is remaining, then generate a new batch with first_step
        if pairs.is_empty() {
            self.0.budget = self.0.budget_min; // Reset budget
            <Sha as BatchOptimizer<_, _, _, _, _, Stepped<_, _, FnState>>>::first_step(self, scp)
        } else {
            // Compute the k to extract top k solution and discard others
            let k = pairs.len() - (((pairs.len() as f64) / self.0.scaling) as usize).max(1);
            self.0.budget = (self.0.budget * self.0.scaling).min(self.0.budget_max); // min to prevent overflowing budget_max
            self.0.iteration += 1;

            // worst solutions before index k, top k  solution at index k and after
            pairs.select_nth_unstable(k);
            // Extract Uncomputed solution from Computed
            let new_pairs: Vec<_> = pairs
                .into_iter()
                .enumerate()
                .map(|(i, computed)| {
                    let (mut pair, _): (Scp::SolShape, _) = IntoComputed::extract(computed);
                    if i < k {
                        pair.discard(); // Discard all solution before k and k others
                    } else {
                        pair.set_fidelity(self.0.budget); // Set new budget for solution after index k
                    }
                    pair
                })
                .collect();

            // If no solution remaining then generate new ones
            if new_pairs.is_empty() {
                self.0.budget = self.0.budget_min; // Reset budget
                <Sha as BatchOptimizer<_, _, _, _, _, Stepped<_, _, FnState>>>::first_step(
                    self, scp,
                )
            } else {
                Batch::new(
                    new_pairs,
                    ShaInfo {
                        iteration: self.0.iteration,
                    }
                    .into(),
                )
            }
        }
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.0.batch = batch_size
    }

    fn get_batch_size(&self) -> usize {
        self.0.batch
    }
}

fn main() {
    let _sha = Sha::new(4, 1.0, 16.0, 2.0);
}
