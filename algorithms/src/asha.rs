//! # Asynchronous Successive Halving
//!
//! ASHA maintains result rungs for a sequence of fidelity budgets. When a
//! worker requests a candidate, it promotes an eligible completed candidate
//! from a lower rung when available; otherwise it asks the inner sampler for
//! a new candidate at the minimum budget. Promotion is based on the
//! single-objective ordering of computed results, and does not wait for every
//! candidate at a rung to finish.
//!
//! [`Asha::new`] takes an inner sampler, minimum budget, maximum budget, and
//! scaling factor. Budgets must be finite and positive, `budget_max` must be
//! greater than or equal to `budget_min`, and scaling must be finite and
//! greater than `1.0`. Invalid budget parameters are returned as
//! [`PrunerError`].
//!
//! [`Asha::into_hyperband`] converts an ASHA instance to the separate
//! `BrackettedAsha` type for multiple brackets. It does not create the
//! [`Hyperband`](crate::Hyperband) type, which wraps SHA.
//!
//! ASHA requires a multi-fidelity problem, an inner sampler that implements
//! [`Prunable`], and computed results that implement [`Orderable`].
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
//! # Reference
//!
//! Li et al., [*A System for Massively Parallel Hyperparameter Tuning*](https://arxiv.org/abs/1810.05934).

use std::{fmt::Debug, marker::PhantomData};

use log::warn;
use tantale_core::{
    HasFidelity, HasInfo, HasState, HasStep, Orderable, Problem, Sampled, Sampler, SolInfo, State,
    Step,
    aliases::{Accum, CompShape, Opt, Shape},
    optimizer::opt::{Prunable, Pruner},
    solution::shape::IntoComputedShape,
    utils::WrappedInfo,
};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, Visitor},
    ser::SerializeStruct,
};
use tantale_macros::fidelity_bounds;

use crate::utils::{
    BracketInfo, PromotableShape, PrunerError, Rungs, build_halving_budgets,
    build_halving_budgets_with_levels,
};

/// A helper macro to simplify the type signature of [`Asha`].
#[macro_export]
macro_rules! asha {
    ($sampler : ident) => {
        Asha<$sampler, _>
    };
    ($sampler : ty) => {
        Asha<$sampler, _>
    };
}

/// A helper macro to simplify the type signature of [`BrackettedAsha`].
#[macro_export]
macro_rules! bracketted_asha {
    ($sampler : ident) => {
        tantale::algos::asha::BrackettedAsha<$sampler, _>
    };
    ($sampler : ty) => {
        tantale::algos::asha::BrackettedAsha<$sampler, _>
    };
}

/// Asynchronous Successive Halving optimizer for multi-fidelity problems.
///
/// # Overview
///
/// ASHA keeps candidate results in fidelity rungs. It returns eligible
/// candidates for promotion when available, and otherwise delegates new
/// candidate generation at the lowest budget to its inner sampler. Unlike
/// [`Sha`](crate::Sha), ASHA does not require a complete batch at each rung.
///
pub struct Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    /// The [`Sampler`] used for generating new candidates when no computed solutions are available for promotion.
    pub sampler: Smpl,
    /// A vector of budget levels corresponding to the halving rounds.
    pub budgets: Vec<f64>,
    /// Scaling factor ($\eta$) by which the budget is multiplied at each stage.
    pub scaling: f64,
    /// A vector of vectors representing the rungs of the Successive Halving process.
    pub rungs: Rungs<P>,
    /// The current budget level index being processed. This is used to track which rung is currently active for promotions and evaluations.
    pub current_budget: f64,
}

impl<P, Smpl> Debug for Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Asha")
            .field("sampler", &self.sampler)
            .field("budgets", &self.budgets)
            .field("scaling", &self.scaling)
            .field("rungs", &self.rungs)
            .field("current_budget", &self.current_budget)
            .finish()
    }
}

impl<P, Smpl> Clone for Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn clone(&self) -> Self {
        Self {
            sampler: self.sampler.clone(),
            budgets: self.budgets.clone(),
            scaling: self.scaling,
            rungs: self.rungs.clone(),
            current_budget: self.current_budget,
        }
    }
}

impl<Smpl, P> State for Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
}

impl<Smpl, P> HasState for Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    type State = Self;

    fn get_mut_state(&mut self) -> &mut Self::State {
        self
    }

    fn get_state(&self) -> &Self::State {
        self
    }

    fn from_state(state: Self::State) -> Self {
        state
    }
}

#[fidelity_bounds(P)]
impl<Smpl, P> Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    /// Creates a new [`Asha`] optimizer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// * `sampler` - Inner sampler used for candidates at the minimum budget.
    /// * `budget_min` - Finite, positive fidelity of the first rung.
    /// * `budget_max` - Finite maximum fidelity, greater than or equal to
    ///   `budget_min`.
    /// * `scaling` - Finite budget multiplier greater than `1.0`.
    ///
    /// # Errors
    ///
    /// Returns an error if the scaling factor or budget range is invalid.
    pub fn new(
        sampler: Smpl,
        budget_min: f64,
        budget_max: f64,
        scaling: f64,
    ) -> Result<Self, PrunerError> {
        let budgets = build_halving_budgets(budget_min, budget_max, scaling)?;
        let rungs = (0..budgets.len()).map(|_| Vec::new()).collect();
        let current_budget = budgets[0];
        Ok(Asha {
            sampler,
            budgets,
            scaling,
            rungs,
            current_budget,
        })
    }

    pub fn into_hyperband(self) -> BrackettedAsha<Smpl, P> {
        let (budget_min, budget_max) = self.get_budgets();
        let sampler = self.sampler;
        let scaling = self.scaling;

        let s_max = (budget_max / budget_min).log(scaling).floor() as usize;
        let num_brackets = s_max + 1;
        let admissions = vec![0; num_brackets];

        let mut max_admissions = Vec::with_capacity(num_brackets);
        let mut budgets = Vec::with_capacity(num_brackets);
        let mut rungs = Vec::with_capacity(num_brackets);

        (0..num_brackets).rev().for_each(|s| {
            let admission =
                (num_brackets as f64 * scaling.powi(s as i32) / ((s + 1) as f64)).ceil() as usize;
            let bmin = budget_max * scaling.powi(-(s as i32));
            let bracket_budgets =
                build_halving_budgets_with_levels(bmin, budget_max, scaling, s + 1).unwrap();
            let bracket_rungs: Rungs<P> = (0..bracket_budgets.len()).map(|_| Vec::new()).collect();
            max_admissions.push(admission);
            budgets.push(bracket_budgets);
            rungs.push(bracket_rungs);
        });

        BrackettedAsha {
            sampler,
            budgets,
            scaling,
            rungs,
            admissions,
            max_admissions,
            num_brackets,
        }
    }
}

#[fidelity_bounds(P, Opt)]
impl<Smpl, P, Opt> Pruner<P, Opt> for Asha<Smpl, P>
where
    Smpl: Sampler<P, Opt> + Prunable,
    CompShape<P>: Orderable,
{
    fn set_budgets(&mut self, budget_min: f64, budget_max: f64) {
        self.budgets = build_halving_budgets(budget_min, budget_max, self.scaling).unwrap();
        self.current_budget = self.budgets[0];
        self.rungs = (0..self.budgets.len()).map(|_| Vec::new()).collect();
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
        assert!(scaling > 1.0, "Scaling factor must be >= 1.0");
        self.scaling = scaling;
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

/// Implementation of the [`Sampler`] trait for Asynchronous Successive Halving.
///
/// Defines the state management and codomain configuration for Asynchronous Successive Halving.
#[fidelity_bounds(P, Opt)]
impl<Smpl, P, Opt> Sampler<P, Opt> for Asha<Smpl, P>
where
    Smpl: Sampler<P, Opt> + Prunable,
    CompShape<P>: Orderable,
{
    type Info = Smpl::Info;

    /// Executes one iteration of Asynchronous Successive Halving on computed candidates.
    ///
    /// This method implements the core algorithm:
    ///
    /// - If a candidate is provided, it is added to the appropriate rung based on its fidelity.
    /// - The optimizer then checks the rungs starting from the highest budget level, selecting the
    ///   top performers and promoting them to the next level of fidelity if the rung has enough candidates.
    /// - If no candidates are available for promotion, it continues down the rungs until
    ///   it finds candidates to promote or defaults to random sampling at the lowest budget level.
    /// - If no candidate is provided, it generates a new random candidate at the lowest budget level.
    ///
    /// Completed candidates are retained in their rungs until promotion or
    /// rung cleanup. The amount of retained state depends on candidate
    /// throughput, rung thresholds, and the number of budget levels.
    ///
    /// In MPI-distributed optimization:
    /// - If there is no Parallel File Sytems, function states and worker states are stored within the worker local memory, and are not shared across workers.
    /// - If there is a Parallel File System, function states and worker states are stored in a shared file system.
    ///
    /// To simplify things, a function state is local to a worker, and is not shared across workers.
    ///
    /// In multi-threaded optimization, function states are stored in the shared memory and can be accessed by all threads.
    fn sample_info(
        &mut self,
        scp: &<P as Problem>::Scp,
        acc: &Accum<P>,
    ) -> Option<(Shape<P>, Sampled<Self::Info>)> {
        let mut i = self.budgets.len() - 1;
        while i > 0 {
            let k = (self.rungs[i].len() as f64 / self.scaling) as usize;
            if k > 0 {
                self.rungs[i].select_nth_unstable_by(k, |a, b| a.ord_cmp(b).unwrap().reverse());
                let mut promotable = self.rungs[i]
                    .iter_mut()
                    .take(k)
                    .filter(|x| !x.is_promoted())
                    .collect::<Vec<_>>();
                if !promotable.is_empty() {
                    promotable.sort_by(|a, b| a.ord_cmp(b).unwrap().reverse());
                    promotable[0].promote();
                    let mut sol: Shape<P> =
                        IntoComputedShape::extract(promotable[0].result.clone()).0;
                    self.current_budget = self.budgets[i];
                    sol.set_fidelity(self.current_budget);
                    return Some((sol, Sampled::Reused));
                }
            }
            i -= 1;
        }

        self.current_budget = self.budgets[0];
        self.sampler.sample_info(scp, acc).map(|(mut sol, info)| {
            sol.set_fidelity(self.current_budget);
            (sol, info)
        })
    }

    fn update(&mut self, x: &CompShape<P>, scp: &<P as Problem>::Scp, acc: &Accum<P>) {
        self.sampler.update(x, scp, acc);
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
                let idx = self.budgets.iter().position(|&b| b == x.fidelity().0);
                if let Some(i) = idx {
                    let rung_idx = i + 1;
                    if rung_idx >= self.rungs.len() {
                        warn!("
                            Warning: Rung index out of bounds due to wrong Step. 
                            Most probably within user-defined objective function.
                            Return Step::Evaluated or Step::Error for fidelity >= budget_max, and Step::Partially for fidelity < budget_max.
                        ");
                    } else {
                        self.rungs[rung_idx].push(PromotableShape::new(x.clone()));
                    }
                }
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
            }
            _ => (),
        }
    }
}

//-------------//
//--- SERDE ---//
//-------------//

struct AshaVisitor<Smpl, P>(PhantomData<(Smpl, P)>)
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>>;

enum AshaField {
    Sampler,
    Budgets,
    Scaling,
    Rung,
    CurrentBudget,
}

struct AshaFieldVisitor;

impl<Smpl, P> Serialize for Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Asha", 5)?;
        state.serialize_field("sampler", self.sampler.get_state())?;
        state.serialize_field("budgets", &self.budgets)?;
        state.serialize_field("scaling", &self.scaling)?;
        state.serialize_field("rung", &self.rungs)?;
        state.serialize_field("current_budget", &self.current_budget)?;
        state.end()
    }
}

impl<Smpl, P> AshaVisitor<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn new() -> Self {
        AshaVisitor(PhantomData)
    }
}

impl<'de> Visitor<'de> for AshaFieldVisitor {
    type Value = AshaField;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("`sampler`, `budgets`, `scaling`, `rung` or `current_budget`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "sampler" => Ok(AshaField::Sampler),
            "budgets" => Ok(AshaField::Budgets),
            "scaling" => Ok(AshaField::Scaling),
            "rung" => Ok(AshaField::Rung),
            "current_budget" => Ok(AshaField::CurrentBudget),
            _ => Err(de::Error::unknown_field(
                value,
                &["sampler", "budgets", "scaling", "rung", "current_budget"],
            )),
        }
    }
}
impl<'de> Deserialize<'de> for AshaField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(AshaFieldVisitor)
    }
}

impl<'de, Smpl, P> Visitor<'de> for AshaVisitor<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    type Value = Asha<Smpl, P>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct Asha")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let sampler = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let budgets = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(1, &self))?;
        let scaling = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(2, &self))?;
        let rung = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(3, &self))?;
        let current_budget = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(4, &self))?;
        Ok(Asha {
            sampler: Smpl::from_state(sampler),
            budgets,
            scaling,
            rungs: rung,
            current_budget,
        })
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: de::MapAccess<'de>,
    {
        let mut sampler = None;
        let mut budgets = None;
        let mut scaling = None;
        let mut rung = None;
        let mut current_budget = None;

        while let Some(key) = map.next_key()? {
            match key {
                AshaField::Sampler => {
                    if sampler.is_some() {
                        return Err(de::Error::duplicate_field("sampler"));
                    }
                    sampler = Some(map.next_value()?);
                }
                AshaField::Budgets => {
                    if budgets.is_some() {
                        return Err(de::Error::duplicate_field("budgets"));
                    }
                    budgets = Some(map.next_value()?);
                }
                AshaField::Scaling => {
                    if scaling.is_some() {
                        return Err(de::Error::duplicate_field("scaling"));
                    }
                    scaling = Some(map.next_value()?);
                }
                AshaField::Rung => {
                    if rung.is_some() {
                        return Err(de::Error::duplicate_field("rung"));
                    }
                    rung = Some(map.next_value()?);
                }
                AshaField::CurrentBudget => {
                    if current_budget.is_some() {
                        return Err(de::Error::duplicate_field("current_budget"));
                    }
                    current_budget = Some(map.next_value()?);
                }
            }
        }

        let sampler = sampler.ok_or_else(|| de::Error::missing_field("sampler"))?;
        let budgets = budgets.ok_or_else(|| de::Error::missing_field("budgets"))?;
        let scaling = scaling.ok_or_else(|| de::Error::missing_field("scaling"))?;
        let rung = rung.ok_or_else(|| de::Error::missing_field("rung"))?;
        let current_budget =
            current_budget.ok_or_else(|| de::Error::missing_field("current_budget"))?;

        Ok(Asha {
            sampler: Smpl::from_state(sampler),
            budgets,
            scaling,
            rungs: rung,
            current_budget,
        })
    }
}

impl<'de, Smpl, P> Deserialize<'de> for Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(AshaVisitor::new())
    }
}

//------------------//
//  BrackettedAsha  //
//------------------//

/// [`BrackettedAsha`] is a variant of the Asynchronous Successive Halving algorithm
/// that organizes the optimization process into multiple brackets,
///  each with its own set of budget levels and rungs.
/// This allows for more flexible and efficient exploration of the search space,
/// as different brackets can operate independently and concurrently, promoting candidates based on their performance within their respective brackets.
pub struct BrackettedAsha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    /// The [`Sampler`] used for generating new candidates when no computed solutions are available for promotion.
    pub sampler: Smpl,
    /// A vector of budget levels corresponding to the halving rounds.
    pub budgets: Vec<Vec<f64>>,
    /// Scaling factor ($\eta$) by which the budget is multiplied at each stage.
    pub scaling: f64,
    /// A vector of vectors of vectors representing the rungs of the Asynchronous Successive Halving process.
    pub rungs: Vec<Rungs<P>>,
    /// The number of admissions per bracket.
    pub admissions: Vec<usize>,
    /// The maximum number of admissions per bracket.
    pub max_admissions: Vec<usize>,
    /// The number of brackets.
    pub num_brackets: usize,
}

impl<P, Smpl> Debug for BrackettedAsha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Asha")
            .field("sampler", &self.sampler)
            .field("budgets", &self.budgets)
            .field("scaling", &self.scaling)
            .field("rungs", &self.rungs)
            .field("admissions", &self.admissions)
            .field("max_admissions", &self.max_admissions)
            .field("num_brackets", &self.num_brackets)
            .finish()
    }
}

impl<P, Smpl> Clone for BrackettedAsha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn clone(&self) -> Self {
        Self {
            sampler: self.sampler.clone(),
            budgets: self.budgets.clone(),
            scaling: self.scaling,
            rungs: self.rungs.clone(),
            admissions: self.admissions.clone(),
            max_admissions: self.max_admissions.clone(),
            num_brackets: self.num_brackets,
        }
    }
}

impl<Smpl, P> State for BrackettedAsha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
}

impl<Smpl, P> HasState for BrackettedAsha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    type State = Self;

    fn get_mut_state(&mut self) -> &mut Self::State {
        self
    }

    fn get_state(&self) -> &Self::State {
        self
    }

    fn from_state(state: Self::State) -> Self {
        state
    }
}

/// Implementation of the [`Sampler`] trait for Asynchronous Successive Halving.
///
/// Defines the state management and codomain configuration for Asynchronous Successive Halving.
#[fidelity_bounds(P, Opt)]
impl<Smpl, P, Opt> Sampler<P, Opt> for BrackettedAsha<Smpl, P>
where
    Smpl: Sampler<P, Opt> + Prunable,
    CompShape<P>: Orderable,
{
    type Info = WrappedInfo<Smpl::Info, BracketInfo>;

    fn sample_info(
        &mut self,
        scp: &<P as Problem>::Scp,
        acc: &Accum<P>,
    ) -> Option<(Shape<P>, Sampled<Self::Info>)> {
        let mut b = 0;
        while b < self.num_brackets {
            let mut i = self.budgets[b].len() - 1;
            while i > 0 {
                let length = self.rungs[b][i].len();
                let k = (length as f64 / self.scaling) as usize;
                if k > 0 {
                    self.rungs[b][i]
                        .select_nth_unstable_by(k, |a, b| a.ord_cmp(b).unwrap().reverse());
                    let mut promotable = self.rungs[b][i]
                        .iter_mut()
                        .take(k)
                        .filter(|x| !x.is_promoted())
                        .collect::<Vec<_>>();
                    if !promotable.is_empty() {
                        promotable.sort_by(|a, b| a.ord_cmp(b).unwrap().reverse());
                        promotable[0].promote();
                        let mut sol: Shape<P> =
                            IntoComputedShape::extract(promotable[0].result.clone()).0;
                        sol.set_fidelity(self.budgets[b][i]);
                        return Some((sol, Sampled::Reused));
                    }
                }
                i -= 1;
            }

            if self.admissions[b] < self.max_admissions[b] {
                self.admissions[b] += 1;
                return self.sampler.sample_info(scp, acc).map(|(mut sol, info)| {
                    sol.set_fidelity(self.budgets[b][0]);
                    let info = match info {
                        Sampled::Fresh(inf) => inf,
                        Sampled::Reused => sol
                            .sinfo()
                            .find_info::<Smpl::Info>()
                            .expect("Sampler info not found in solution info")
                            .clone(),
                    };
                    let sinfo = WrappedInfo::new(info, BracketInfo(b).into());
                    (sol, Sampled::Fresh(sinfo.into()))
                });
            }
            b += 1;
        }
        self.admissions = vec![0; self.num_brackets];
        self.rungs.iter_mut().for_each(|bracket| {
            bracket.iter_mut().for_each(|rung| {
                rung.drain(..);
            })
        });
        self.sample_info(scp, acc)
    }

    fn update(&mut self, x: &CompShape<P>, scp: &<P as Problem>::Scp, acc: &Accum<P>) {
        self.sampler.update(x, scp, acc);
        match x.step() {
            Step::Partially(s) => {
                let fid = x.fidelity().0;
                let info = x
                    .sinfo()
                    .find_info::<WrappedInfo<Smpl::Info, BracketInfo>>()
                    .expect("BracketInfo not found in solution info");
                let bracket = info.super_info.0;
                let max_budget = *self.budgets[bracket].last().unwrap();
                if fid >= max_budget {
                    warn!(
                        "
                            Fidelity {} >= budget_max {} for a Partially({}) step.
                            This may indicate a logic error in the objective function.
                        ",
                        fid, max_budget, s
                    );
                }

                let idx = self.budgets[bracket]
                    .iter()
                    .position(|&b| b == x.fidelity().0);
                if let Some(i) = idx {
                    let rung_idx = i + 1;
                    if rung_idx >= self.rungs[bracket].len() {
                        warn!("
                            Warning: Rung index out of bounds due to wrong Step. 
                            Most probably within user-defined objective function.
                            Return Step::Evaluated or Step::Error for fidelity >= budget_max, and Step::Partially for fidelity < budget_max.
                        ");
                    } else {
                        self.rungs[bracket][rung_idx].push(PromotableShape::new(x.clone()));
                    }
                }
            }
            Step::Evaluated => {
                let fid = x.fidelity().0;
                let info = x
                    .sinfo()
                    .find_info::<WrappedInfo<Smpl::Info, BracketInfo>>()
                    .expect("BracketInfo not found in solution info");
                let bracket = info.super_info.0;
                let max_budget = *self.budgets[bracket].last().unwrap();
                if fid >= max_budget {
                    warn!(
                        "
                            Fidelity {} < budget_max {} for an Evaluated step.
                            This may indicate a logic error in the objective function.
                        ",
                        fid, max_budget
                    );
                }
            }
            _ => (),
        }
    }
}

//-------------//
//--- SERDE ---//
//-------------//

struct BrackettedAshaVisitor<Smpl, P>(PhantomData<(Smpl, P)>)
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>>;

enum BrackettedAshaField {
    Sampler,
    Budgets,
    Scaling,
    Rung,
    Admissions,
    MaxAdmissions,
    NumBrackets,
}

struct BrackettedAshaFieldVisitor;

impl<Smpl, P> Serialize for BrackettedAsha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Asha", 7)?;
        state.serialize_field("sampler", self.sampler.get_state())?;
        state.serialize_field("budgets", &self.budgets)?;
        state.serialize_field("scaling", &self.scaling)?;
        state.serialize_field("rung", &self.rungs)?;
        state.serialize_field("admissions", &self.admissions)?;
        state.serialize_field("max_admissions", &self.max_admissions)?;
        state.serialize_field("num_brackets", &self.num_brackets)?;
        state.end()
    }
}

impl<Smpl, P> BrackettedAshaVisitor<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn new() -> Self {
        BrackettedAshaVisitor(PhantomData)
    }
}

impl<'de> Visitor<'de> for BrackettedAshaFieldVisitor {
    type Value = BrackettedAshaField;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("`sampler`, `budgets`, `scaling`, `rung`, `admissions`, `max_admissions` or `num_brackets`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "sampler" => Ok(BrackettedAshaField::Sampler),
            "budgets" => Ok(BrackettedAshaField::Budgets),
            "scaling" => Ok(BrackettedAshaField::Scaling),
            "rung" => Ok(BrackettedAshaField::Rung),
            "admissions" => Ok(BrackettedAshaField::Admissions),
            "max_admissions" => Ok(BrackettedAshaField::MaxAdmissions),
            "num_brackets" => Ok(BrackettedAshaField::NumBrackets),
            _ => Err(de::Error::unknown_field(
                value,
                &[
                    "sampler",
                    "budgets",
                    "scaling",
                    "rung",
                    "admissions",
                    "max_admissions",
                    "num_brackets",
                ],
            )),
        }
    }
}
impl<'de> Deserialize<'de> for BrackettedAshaField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(BrackettedAshaFieldVisitor)
    }
}

impl<'de, Smpl, P> Visitor<'de> for BrackettedAshaVisitor<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    type Value = BrackettedAsha<Smpl, P>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct Asha")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let sampler = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let budgets = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(1, &self))?;
        let scaling = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(2, &self))?;
        let rung = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(3, &self))?;
        let admissions = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(4, &self))?;
        let max_admissions = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(5, &self))?;
        let num_brackets = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(6, &self))?;
        Ok(BrackettedAsha {
            sampler: Smpl::from_state(sampler),
            budgets,
            scaling,
            rungs: rung,
            admissions,
            max_admissions,
            num_brackets,
        })
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: de::MapAccess<'de>,
    {
        let mut sampler = None;
        let mut budgets = None;
        let mut scaling = None;
        let mut rung = None;
        let mut admissions = None;
        let mut max_admissions = None;
        let mut num_brackets = None;

        while let Some(key) = map.next_key()? {
            match key {
                BrackettedAshaField::Sampler => {
                    if sampler.is_some() {
                        return Err(de::Error::duplicate_field("sampler"));
                    }
                    sampler = Some(map.next_value()?);
                }
                BrackettedAshaField::Budgets => {
                    if budgets.is_some() {
                        return Err(de::Error::duplicate_field("budgets"));
                    }
                    budgets = Some(map.next_value()?);
                }
                BrackettedAshaField::Scaling => {
                    if scaling.is_some() {
                        return Err(de::Error::duplicate_field("scaling"));
                    }
                    scaling = Some(map.next_value()?);
                }
                BrackettedAshaField::Rung => {
                    if rung.is_some() {
                        return Err(de::Error::duplicate_field("rung"));
                    }
                    rung = Some(map.next_value()?);
                }
                BrackettedAshaField::Admissions => {
                    if admissions.is_some() {
                        return Err(de::Error::duplicate_field("admissions"));
                    }
                    admissions = Some(map.next_value()?);
                }
                BrackettedAshaField::MaxAdmissions => {
                    if max_admissions.is_some() {
                        return Err(de::Error::duplicate_field("max_admissions"));
                    }
                    max_admissions = Some(map.next_value()?);
                }
                BrackettedAshaField::NumBrackets => {
                    if num_brackets.is_some() {
                        return Err(de::Error::duplicate_field("num_brackets"));
                    }
                    num_brackets = Some(map.next_value()?);
                }
            }
        }

        let sampler = sampler.ok_or_else(|| de::Error::missing_field("sampler"))?;
        let budgets = budgets.ok_or_else(|| de::Error::missing_field("budgets"))?;
        let scaling = scaling.ok_or_else(|| de::Error::missing_field("scaling"))?;
        let rung = rung.ok_or_else(|| de::Error::missing_field("rung"))?;
        let admissions = admissions.ok_or_else(|| de::Error::missing_field("admissions"))?;
        let max_admissions =
            max_admissions.ok_or_else(|| de::Error::missing_field("max_admissions"))?;
        let num_brackets = num_brackets.ok_or_else(|| de::Error::missing_field("num_brackets"))?;

        Ok(BrackettedAsha {
            sampler: Smpl::from_state(sampler),
            budgets,
            scaling,
            rungs: rung,
            admissions,
            max_admissions,
            num_brackets,
        })
    }
}

impl<'de, Smpl, P> Deserialize<'de> for BrackettedAsha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(BrackettedAshaVisitor::new())
    }
}
