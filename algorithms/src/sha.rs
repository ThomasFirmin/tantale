//! Successive Halving algorithm for multi-fidelity hyperparameter optimization.
//!
//! # Overview
//!
//! Successive Halving is a multi-fidelity optimization strategy that iteratively:
//! 1. Evaluates a batch of candidates at varying resource levels (fidelity)
//! 2. Selects the Top-k performing candidates
//! 3. Discards poorly-performing ones
//! 4. Re-evaluates remaining candidates at higher fidelity levels
//!
//! This approach balances exploration and exploitation by spending more computational
//! resources on promising candidates. It is particularly effective for expensive optimization
//! problems where evaluation cost scales with fidelity.
//!
//! The budget is increased by a factor of `scaling` in each successive stage until reaching `budget_max`,
//! then the process restarts with `budget_min`.
//!
//! # Pseudo-code
//!
//! **Successive Halving (SH)**
//! ---
//! **Inputs**
//! 1. &emsp; $\mathcal{X}$ &emsp;&emsp; *A [searchspace](tantale_core::Searchspace)*
//! 2. &emsp; $n$ &emsp;&emsp; *[batch](tantale_core::Batch) size, s.t. $n \geq \eta^{\left\lfloor\log_\eta(b_\text{max}/b_0)\right\rfloor}$*
//! 3. &emsp; $b_0$ &emsp;&emsp; *Initial budget*
//! 4. &emsp; $b_{\text{max}}$ &emsp;&emsp; *Maximum budget*
//! 5. &emsp; $\eta$ &emsp;&emsp; *Scaling*
//! 6. &emsp;
//! 7. &emsp; $b \gets b_0$
//! 8. &emsp; $L \gets \text{Random}(\mathcal{X},n)$ &emsp; *Sample $n$ [solution](tantale_core::Solution)s*
//! 9. &emsp; **while** $b < b_\text{max}$ **do**
//! 10. &emsp; &emsp; $\mathbf{y} \gets f(L;b)$ &emsp; *Evaluate $L$ with [fidelity](tantale_core::Fidelity) $b$*
//! 11. &emsp; &emsp; $L \gets \text{Top}_k\left(L,\mathbf{y},\left\lfloor \frac{\lvert L \rvert }{\eta} \right\rfloor\right)$ *Select the top $\left\lfloor \frac{\lvert L \rvert }{\eta} \right\rfloor$ best [computed](tantale_core::Computed)*
//! 12. &emsp; &emsp; $b \gets \eta \times b$
//! 13. &emsp; **return best of $(L,\mathbf{y})$**
//! ---
//! 
//! SHA evaluates a fixed-size batch at each budget level. After the batch is
//! computed, it ranks the candidates, promotes the best candidates to the next
//! budget, and discards the rest. The budget sequence starts at `budget_min`,
//! grows by `scaling`, and includes `budget_max` exactly. After the final
//! budget rung, the sampler starts a new batch at the minimum budget.
//! 
//! # Type Parameters
//!
//! The inner sampler proposes candidates for the first rung. SHA requires
//! comparable computed solutions and a multi-fidelity problem whose outcome
//! reports step completion and whose function accepts fidelity.
//!
//! # Note
//!
//! SHA is a sampler/pruner used by a core experiment. The experiment's stop
//! criterion controls the total run; completing a final rung starts another
//! batch unless the experiment has stopped.
//!
//! # References
//!
//! Successive Halving is based on the work of [Li et al. (2018)](https://arxiv.org/pdf/1810.05934).

use std::{fmt::Debug, marker::PhantomData};

use log::warn;
use tantale_core::{
    HasFidelity, HasState, HasStep, Orderable, Problem, Sampled, Sampler, State, Step,
    aliases::{Accum, CompShape, Opt, Shape},
    optimizer::opt::{Batched, Prunable, Pruner},
    solution::shape::IntoComputedShape,
    utils::ShapeWithInfo,
};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, Visitor},
    ser::SerializeStruct,
};
use tantale_macros::fidelity_bounds;

use crate::{
    Hyperband,
    utils::{PrunerError, build_halving_budgets, refill_apply},
};

/// A helper macro to simplify the type signature of [`Sha`].
#[macro_export]
macro_rules! sha {
    ($sampler : ident) => {
        Sha<$sampler, _>
    };
    ($sampler : ty) => {
        Sha<$sampler, _>
    };
}

/// Successive Halving multi-fidelity optimizer.
///
/// A [`Sampler`] implementing the Successive Halving algorithm
/// for multi-fidelity evaluations.
///
/// # Overview
///
/// Synchronous Successive Halving optimizer for multi-fidelity evaluations.
///
/// SHA collects a full rung of evaluated candidates before ranking and
/// promoting candidates. The `sampler` supplies candidates at the first rung;
/// the `batch_size` is the number of candidates in that initial rung. Later
/// rung sizes are reduced according to `scaling`. A completed final rung is
/// followed by a fresh initial rung.
///
/// The problem's computed solutions must implement [`Orderable`], and the
/// sampler must implement [`Prunable`].
pub struct Sha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    /// The sampler used for generating new candidates when no computed solutions are available for promotion.
    pub sampler: Smpl,
    /// The batch size used for sampling candidates.
    pub batch_size: usize,
    /// The current batch of candidates.
    pub batch: Vec<ShapeWithInfo<Shape<P>, Smpl::Info>>,
    /// The current batch of computed candidates.
    pub comp_batch: Vec<CompShape<P>>,
    /// A vector of budget levels corresponding to the halving rounds.
    pub budgets: Vec<f64>,
    /// Current budget level. This value increases by `scaling` at each stage until it reaches `budget_max`.
    pub current_budget: f64,
    /// Size of the current rung (number of candidates at the current budget level).
    pub current_rung_size: usize,
    /// Index of the current budget in the `budgets` vector.
    pub budget_idx: usize,
    /// Scaling factor ($\eta$) by which the budget is multiplied at each stage.
    /// It must be finite and greater than `1.0`.
    pub scaling: f64,
    /// Count of pending candidates that have been sampled but not yet evaluated.
    ///  This is used to track the number of candidates in the current batch that are awaiting evaluation.
    pub pending: usize,
}

impl<P, Smpl> Clone for Sha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn clone(&self) -> Self {
        Sha {
            sampler: self.sampler.clone(),
            batch_size: self.batch_size,
            batch: self.batch.clone(),
            comp_batch: self.comp_batch.clone(),
            budgets: self.budgets.clone(),
            current_budget: self.current_budget,
            current_rung_size: self.current_rung_size,
            budget_idx: self.budget_idx,
            scaling: self.scaling,
            pending: 0,
        }
    }
}

impl<P, Smpl> Debug for Sha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sha")
            .field("sampler", &self.sampler)
            .field("batch_size", &self.batch_size)
            .field("batch", &self.batch)
            .field("comp_batch", &self.comp_batch)
            .field("budgets", &self.budgets)
            .field("current_budget", &self.current_budget)
            .field("current_rung_size", &self.current_rung_size)
            .field("budget_idx", &self.budget_idx)
            .field("scaling", &self.scaling)
            .field("pending", &self.pending)
            .finish()
    }
}

#[fidelity_bounds(P)]
impl<Smpl, P> Sha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    /// Creates a new Successive Halving optimizer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// * `sampler` - Sampler used to create candidates for the first rung.
    /// * `budget_min` - Finite, positive fidelity of the first rung.
    /// * `budget_max` - Finite maximum fidelity, greater than or equal to
    ///   `budget_min`.
    /// * `scaling` - Finite budget multiplier greater than `1.0`.
    /// * `batch_size` - Number of candidates in the initial rung.
    ///
    /// # Errors
    ///
    /// Returns an error when the scaling factor or budget range is invalid.
    /// The batch size is not validated.
    pub fn new(
        sampler: Smpl,
        budget_min: f64,
        budget_max: f64,
        scaling: f64,
        batch_size: usize,
    ) -> Result<Self, PrunerError> {
        let budgets = build_halving_budgets(budget_min, budget_max, scaling)?;
        let current_rung_size = batch_size;
        let current_budget = budgets[0];
        Ok(Sha {
            sampler,
            batch_size,
            batch: Vec::new(),
            comp_batch: Vec::new(),
            budgets,
            budget_idx: 0,
            current_budget,
            current_rung_size,
            scaling,
            pending: 0,
        })
    }

    /// Returns `true` if the current rung is complete, 
    /// meaning that the current budget has reached the maximum budget, 
    /// the batch is empty, and the number of computed candidates is at least equal to the current rung size.
    pub fn bracket_complete(&self) -> bool {
        self.current_budget == *self.budgets.last().unwrap()
            && self.batch.is_empty()
            && self.comp_batch.len() >= self.current_rung_size
            && self.pending == 0
    }

    /// Converts the current SHA instance into a [`Hyperband`] instance, which manages multiple brackets of Successive Halving.
    pub fn into_hyperband(mut self) -> Hyperband<Smpl, P> {
        let (budget_min, budget_max) = self.get_budgets();
        let scaling = self.get_scaling();

        let s_max = (budget_max / budget_min).log(scaling).floor() as usize;

        let n = scaling.powi(s_max as i32).ceil() as usize;
        self.set_batch_size(n);
        let r = budget_max * scaling.powi(-(s_max as i32));
        self.set_budgets(r, budget_max);

        Hyperband::new(budget_min, budget_max, scaling, s_max, self)
    }
}

impl<Smpl, P> State for Sha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
}

impl<Smpl, P> HasState for Sha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
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

#[fidelity_bounds(P, Opt)]
impl<Smpl, P, Opt> Batched<P, Opt> for Sha<Smpl, P>
where
    Smpl: Sampler<P, Opt> + Prunable,
    CompShape<P>: Orderable,
{
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Sets the batch size for the optimizer. This affects how many candidates are sampled in each iteration.
    /// Resetting the batch size will also reset the current budget and rung size to the initial values,
    /// and clear any existing candidates in the batch.
    fn set_batch_size(&mut self, size: usize) {
        self.batch_size = size;
        self.current_rung_size = size;
        self.current_budget = self.budgets[0];
        self.budget_idx = 0;
        self.batch.clear();
        self.comp_batch.clear();
    }

    fn batch(&self) -> &[ShapeWithInfo<Shape<P>, Smpl::Info>] {
        &self.batch
    }

    fn mut_batch(&mut self) -> &mut Vec<ShapeWithInfo<Shape<P>, Smpl::Info>> {
        &mut self.batch
    }

    fn comp_batch(&self) -> &[CompShape<P>] {
        &self.comp_batch
    }

    fn mut_comp_batch(&mut self) -> &mut Vec<CompShape<P>> {
        &mut self.comp_batch
    }
}

#[fidelity_bounds(P, Opt)]
impl<Smpl, P, Opt> Pruner<P, Opt> for Sha<Smpl, P>
where
    Smpl: Sampler<P, Opt> + Prunable,
    CompShape<P>: Orderable,
{
    /// Reinitializes the budget parameters for this optimizer.
    /// This can be used to adjust the fidelity levels during optimization or before restarting a new run
    /// Rungs are cleared when budgets are updated, as the previous candidates may not be relevant to the new budget configuration.
    fn set_budgets(&mut self, budget_min: f64, budget_max: f64) {
        self.budgets = build_halving_budgets(budget_min, budget_max, self.scaling).unwrap();
        self.current_budget = self.budgets[0];
        self.budget_idx = 0;
        self.batch.clear();
        self.comp_batch.clear();
    }

    /// Returns the current minimum and maximum budgets of this optimizer.
    fn get_budgets(&self) -> (f64, f64) {
        (
            *self.budgets.first().unwrap(),
            *self.budgets.last().unwrap(),
        )
    }

    /// Updates the scaling factor for this optimizer.
    fn set_scaling(&mut self, scaling: f64) {
        assert!(scaling > 1.0, "Scaling factor must be > 1.0");
        self.scaling = scaling;
        self.budgets =
            build_halving_budgets(self.get_budget_min(), self.get_budget_max(), self.scaling)
                .unwrap();
        self.current_budget = self.budgets[0];
        self.budget_idx = 0;
        self.batch.clear();
        self.comp_batch.clear();
    }

    /// Returns the current scaling factor for this optimizer.
    fn get_scaling(&self) -> f64 {
        self.scaling
    }

    /// Sets the current budget level used by the optimizer for pruning candidates.
    fn set_current_budget(&mut self, budget: f64) {
        assert!(
            budget >= self.budgets[0] && budget <= *self.budgets.last().unwrap(),
            "Current budget must be within the range of defined budgets"
        );
        self.current_budget = budget;
    }

    fn get_current_budget(&self) -> f64 {
        self.current_budget
    }
}

#[fidelity_bounds(P, Opt)]
impl<Smpl, P, Opt> Sampler<P, Opt> for Sha<Smpl, P>
where
    Smpl: Sampler<P, Opt> + Prunable,
    CompShape<P>: Orderable,
{
    type Info = Smpl::Info;

    fn sample_info(
        &mut self,
        scp: &<P as Problem>::Scp,
        acc: &Accum<P>,
    ) -> Option<(Shape<P>, Sampled<Self::Info>)> {
        if let Some(sol) = self.batch.pop() {
            match sol.0.step() {
                Step::Discard | Step::Evaluated => (),
                _ => self.pending += 1,
            }
            Some((sol.0, sol.1))
        } else if self.pending > 0 {
            None // Wait for more computed solutions to be available before promoting to the next rung
        } else {
            self.comp_batch
                .retain(|comp| matches!(comp.step(), Step::Partially(_)));
            if self.comp_batch().is_empty() {
                self.budget_idx = 0;
                self.current_budget = self.budgets[0];
                self.current_rung_size = self.batch_size;
                while self.batch.len() < self.current_rung_size
                    && let Some(mut sol) = self.sampler.sample_info(scp, acc)
                {
                    sol.0.set_fidelity(self.current_budget);
                    self.batch.push(ShapeWithInfo(sol.0, sol.1));
                }
                if self.batch.len() < self.current_rung_size {
                    warn!(
                        "
                            Insufficient candidates sampled: expected {}, got {}. Consider investigating the sampler.
                            The rung will be filled with randomly sampled candidates, but this may affect the optimization performance.
                        ",
                        self.current_rung_size,
                        self.batch.len()
                    );
                    refill_apply::<P, _, _>(
                        &mut self.batch,
                        scp,
                        self.current_rung_size,
                        |mut sol| {
                            sol.set_fidelity(self.current_budget);
                            (sol, Smpl::Info::default())
                        },
                    );
                }
                self.sample_info(scp, acc)
            } else {
                // Advance to the next fidelity rung.
                self.budget_idx += 1;
                self.current_budget = self.budgets[self.budget_idx];
                self.current_rung_size = ((self.batch_size() as f64
                    * self.scaling.powi(-(self.budget_idx as i32)))
                .floor() as usize)
                    .max(1);

                let comp_len = self.mut_comp_batch().len();
                let survivors = self.current_rung_size.min(comp_len);

                let discard_count = comp_len - survivors;

                // Put the worst candidates first and the best candidates last.
                self.mut_comp_batch()
                    .select_nth_unstable_by(discard_count, |a, b| {
                        Orderable::ord_cmp(a, b).unwrap()
                    });

                let mut promoted = 0;

                while let Some(comp) = self.mut_comp_batch().pop() {
                    let mut pair: Shape<P> = IntoComputedShape::extract(comp).0;

                    if promoted < survivors {
                        pair.set_fidelity(self.current_budget);
                        self.mut_batch().push(ShapeWithInfo(pair, Sampled::Reused));
                        promoted += 1;
                    } else {
                        pair.discard();
                        self.mut_batch().push(ShapeWithInfo(pair, Sampled::Reused));
                    }
                }
                self.sample_info(scp, acc)
            }
        }
    }

    fn update(&mut self, x: &CompShape<P>, scp: &<P as Problem>::Scp, acc: &Accum<P>) {
        self.pending -= 1;
        match x.step() {
            Step::Partially(s) => {
                let fid = x.fidelity().0;
                if fid >= self.get_budget_max() {
                    warn!(
                        "
                            Fidelity {} >= budget_max {} for a Partially({}) step.
                            This may indicate a logic error in the objective function.
                        ",
                        fid,
                        self.get_budget_max(),
                        s
                    );
                }
                self.comp_batch.push(x.clone());
                self.sampler.update(x, scp, acc)
            }
            Step::Evaluated => {
                let fid = x.fidelity().0;
                if fid < self.get_budget_max() {
                    warn!(
                        "
                            Fidelity {} < budget_max {} for an Evaluated step.
                            This may indicate a logic error in the objective function.
                        ",
                        fid,
                        self.get_budget_max()
                    );
                }
                self.comp_batch.push(x.clone());
                self.sampler.update(x, scp, acc)
            }
            _ => {
                warn!(
                    "
                        Received a Step::{:?} for a candidate in Successive Halving.
                        This is unexpected, as only Partially or Evaluated steps should be updated.
                    ",
                    x.step()
                );
            }
        }
    }
}

//-------------//
//--- SERDE ---//
//-------------//

struct ShaVisitor<Smpl, P>(PhantomData<(Smpl, P)>)
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable;

enum ShaField {
    Sampler,
    BatchSize,
    Batch,
    CompBatch,
    Budgets,
    CurrentBudget,
    CurrentRungSize,
    BudgetIdx,
    Scaling,
    Refilled,
}

struct ShaFieldVisitor;

impl<Smpl, P> Serialize for Sha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ShaState", 10)?;
        state.serialize_field("sampler", &self.sampler.get_state())?;
        state.serialize_field("batch_size", &self.batch_size)?;
        state.serialize_field("batch", &self.batch)?;
        state.serialize_field("comp_batch", &self.comp_batch)?;
        state.serialize_field("budgets", &self.budgets)?;
        state.serialize_field("current_budget", &self.current_budget)?;
        state.serialize_field("current_rung_size", &self.current_rung_size)?;
        state.serialize_field("budget_idx", &self.budget_idx)?;
        state.serialize_field("scaling", &self.scaling)?;
        state.serialize_field("pending", &self.pending)?;
        state.end()
    }
}

impl<Smpl, P> ShaVisitor<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn new() -> Self {
        ShaVisitor(PhantomData)
    }
}

impl<'de> Visitor<'de> for ShaFieldVisitor {
    type Value = ShaField;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter
            .write_str(
                "'sampler', 'batch_size', 'batch', 'comp_batch', 'budgets', 'current_budget', 'current_rung_size', 'budget_idx', 'scaling', 'pending'"
            )
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "sampler" => Ok(ShaField::Sampler),
            "batch_size" => Ok(ShaField::BatchSize),
            "batch" => Ok(ShaField::Batch),
            "comp_batch" => Ok(ShaField::CompBatch),
            "budgets" => Ok(ShaField::Budgets),
            "current_budget" => Ok(ShaField::CurrentBudget),
            "current_rung_size" => Ok(ShaField::CurrentRungSize),
            "budget_idx" => Ok(ShaField::BudgetIdx),
            "scaling" => Ok(ShaField::Scaling),
            "pending" => Ok(ShaField::Refilled),
            _ => Err(de::Error::unknown_field(
                value,
                &[
                    "sampler",
                    "batch_size",
                    "batch",
                    "comp_batch",
                    "budgets",
                    "current_budget",
                    "current_rung_size",
                    "budget_idx",
                    "scaling",
                    "pending",
                ],
            )),
        }
    }
}
impl<'de> Deserialize<'de> for ShaField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(ShaFieldVisitor)
    }
}

impl<'de, Smpl, P> Visitor<'de> for ShaVisitor<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    type Value = Sha<Smpl, P>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct ShaState")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let sampler = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let batch_size = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(1, &self))?;
        let batch = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(2, &self))?;
        let comp_batch = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(3, &self))?;
        let budgets = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(4, &self))?;
        let current_budget = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(5, &self))?;
        let current_rung_size = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(6, &self))?;
        let budget_idx = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(7, &self))?;
        let scaling = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(8, &self))?;
        let pending = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(9, &self))?;

        Ok(Sha {
            sampler: Smpl::from_state(sampler),
            batch_size,
            batch,
            comp_batch,
            budgets,
            current_budget,
            current_rung_size,
            budget_idx,
            scaling,
            pending,
        })
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: de::MapAccess<'de>,
    {
        let mut sampler = None;
        let mut batch_size = None;
        let mut batch = None;
        let mut comp_batch = None;
        let mut budgets = None;
        let mut current_budget = None;
        let mut current_rung_size = None;
        let mut budget_idx = None;
        let mut scaling = None;
        let mut pending = None;

        while let Some(key) = map.next_key()? {
            match key {
                ShaField::Sampler => {
                    if sampler.is_some() {
                        return Err(de::Error::duplicate_field("sampler"));
                    }
                    sampler = Some(map.next_value()?);
                }
                ShaField::BatchSize => {
                    if batch_size.is_some() {
                        return Err(de::Error::duplicate_field("batch_size"));
                    }
                    batch_size = Some(map.next_value()?);
                }
                ShaField::Batch => {
                    if batch.is_some() {
                        return Err(de::Error::duplicate_field("batch"));
                    }
                    batch = Some(map.next_value()?);
                }
                ShaField::CompBatch => {
                    if comp_batch.is_some() {
                        return Err(de::Error::duplicate_field("comp_batch"));
                    }
                    comp_batch = Some(map.next_value()?);
                }
                ShaField::Budgets => {
                    if budgets.is_some() {
                        return Err(de::Error::duplicate_field("budgets"));
                    }
                    budgets = Some(map.next_value()?);
                }
                ShaField::CurrentBudget => {
                    if current_budget.is_some() {
                        return Err(de::Error::duplicate_field("current_budget"));
                    }
                    current_budget = Some(map.next_value()?);
                }
                ShaField::CurrentRungSize => {
                    if current_rung_size.is_some() {
                        return Err(de::Error::duplicate_field("current_rung_size"));
                    }
                    current_rung_size = Some(map.next_value()?);
                }
                ShaField::BudgetIdx => {
                    if budget_idx.is_some() {
                        return Err(de::Error::duplicate_field("budget_idx"));
                    }
                    budget_idx = Some(map.next_value()?);
                }
                ShaField::Scaling => {
                    if scaling.is_some() {
                        return Err(de::Error::duplicate_field("scaling"));
                    }
                    scaling = Some(map.next_value()?);
                }
                ShaField::Refilled => {
                    if pending.is_some() {
                        return Err(de::Error::duplicate_field("pending"));
                    }
                    pending = Some(map.next_value()?);
                }
            }
        }

        let sampler = sampler.ok_or_else(|| de::Error::missing_field("sampler"))?;
        let batch_size = batch_size.ok_or_else(|| de::Error::missing_field("batch_size"))?;
        let batch = batch.ok_or_else(|| de::Error::missing_field("batch"))?;
        let comp_batch = comp_batch.ok_or_else(|| de::Error::missing_field("comp_batch"))?;
        let budgets = budgets.ok_or_else(|| de::Error::missing_field("budgets"))?;
        let current_budget =
            current_budget.ok_or_else(|| de::Error::missing_field("current_budget"))?;
        let current_rung_size =
            current_rung_size.ok_or_else(|| de::Error::missing_field("current_rung_size"))?;
        let budget_idx = budget_idx.ok_or_else(|| de::Error::missing_field("budget_idx"))?;
        let scaling = scaling.ok_or_else(|| de::Error::missing_field("scaling"))?;
        let pending = pending.ok_or_else(|| de::Error::missing_field("pending"))?;

        Ok(Sha {
            sampler: Smpl::from_state(sampler),
            batch_size,
            batch,
            comp_batch,
            budgets,
            current_budget,
            current_rung_size,
            budget_idx,
            scaling,
            pending,
        })
    }
}

impl<'de, Smpl, P> Deserialize<'de> for Sha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(ShaVisitor::new())
    }
}
