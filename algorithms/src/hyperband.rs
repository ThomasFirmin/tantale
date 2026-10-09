//! [Hyperband](https://arxiv.org/pdf/1603.06560) algorithm for multi-fidelity hyperparameter optimization.
//!
//! # Overview
//!
//! Hyperband is a bandit-based algorithm that manages multiple brackets of successive halving runs
//! in parallel. It efficiently allocates a fixed computational budget across brackets to discover
//! high-performing configurations. Each bracket progressively increases fidelity while eliminating
//! poorly-performing candidates through successive halving.
//!
//! # Pseudo-code
//!
//! **Hyperband**
//! ---
//! **Inputs**
//! 1. &emsp; $\mathcal{O}$ &emsp;&emsp; *An inner optimizer implementing [`BudgetPruner`] (e.g., SHA or ASHA)*
//! 2. &emsp; $b_{\min}$ &emsp;&emsp; *Minimum budget*
//! 3. &emsp; $b_{\max}$ &emsp;&emsp; *Maximum budget*
//! 4. &emsp; $\eta$ &emsp;&emsp; *Scaling factor (from inner optimizer)*
//! 5. &emsp;
//! 6. &emsp; $s_{\max} \gets \lfloor \log_\eta(b_{\max}/b_{\min}) \rfloor$ &emsp; *Maximum bracket index*
//! 7. &emsp;
//! 8. &emsp; **for** $s \in [s_{\max}, \ldots, 0]$ **do**
//! 9. &emsp;  &emsp; **set_max_budget**$(\mathcal{O}, b_{\max}\eta^{-s})$ &emsp; *Configure inner optimizer budgets*
//! 10. &emsp; &emsp; ------ If $\mathcal{O}$ is a [`BatchOptimizer`] ------
//! 11. &emsp; &emsp; **set_batch_size**$\left(\mathcal{O}, \left\lceil \frac{s_{\max}+1}{s+1} \eta^s \right\rceil\right)$ &emsp; *Configure inner optimizer batch size*
//! 12. &emsp; &emsp; ------------------------------------------------------------
//! 13. &emsp; &emsp; **Run** $\mathcal{O}$ until **current_budget**$(\mathcal{O}) \geq b_{\max}\eta^{-s}$ &emsp;
//! 14. &emsp;
//! 15. &emsp; **return best configuration across all brackets**
//! ---
//!
//! # Type Parameters
//!
//! The algorithm is generic over:
//! - **Inner Optimizer**: Must satisfy both [`Optimizer`] and [`BudgetPruner`]
//!   traits. Typically [`Sha`](crate::Sha) or [`Asha`](crate::Asha).
//! - **Output Type**: Must satisfy [`FidOutcome`] for multi-fidelity support
//! - **Searchspace**: Must generate [`SolutionShape`](tantale_core::SolutionShape) with [`HasFidelity`] and [`HasStep`]
//! - **Solution Info**: Must satisfy [`SolInfo`] constraint for the inner optimizer
//!
//! # Example
//!
//! ```ignore
//! // Create inner optimizer (e.g., SHA)
//! let sampler = RandomSearch::new();
//! let sha = Sha::new(
//!     sampler,
//!     batch_size,
//!     budget_min,
//!     budget_max,
//!     scaling,
//! );
//!
//! // Wrap in Hyperband to run multiple brackets
//! let hyperband = Hyperband::new(sha);
//! ```
//!
//! # Bracket Management
//!
//! Each bracket is executed sequentially, with the inner optimizer managing successive halving
//! within each bracket. For [`BatchOptimizer`] inner optimizers (like SHA), brackets run one after another. For
//! [`SingleOptimizer`] (like ASHA) when the inner [`BudgetPruner`] reaches its budget limit,
//! the internal solutions are drained and [`Discarded`](tantale_core::Step::Discard), incoming solution from
//! previous brackets might be included within the next bracket if their fidelity level correspond to new fidelity
//! level of the next bracket.
//!
//! # Notes
//!
//! - Unlike single-bracket algorithms (SHA), Hyperband automatically discovers good initial batch sizes
//! - The algorithm resets the inner optimizer for each bracket to maintain independence
//! - The inner optimizer's state is preserved across checkpoints as part of the [`HyperbandState`]
//! - For batch optimizers (like SHA), brackets execute sequentially; for sequential optimizers
//!   (like ASHA), the implementation determines parallelization strategy
//!
//! The parallelization strategy is the one briefly described in [Li et al. (2018)](https://arxiv.org/pdf/1810.05934).
//!
//! # References
//!
//! Hyperband is based on the work of [Li et al. (2018)](https://arxiv.org/pdf/1603.06560).
//! See also the work on Successive Halving by [Li et al. (2020)](https://arxiv.org/pdf/1810.05934).

use std::marker::PhantomData;

use serde::de::{self, Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};
use tantale_core::aliases::{Accum, CompShape, Shape};
use tantale_core::optimizer::opt::Pruner;
use tantale_core::solution::shape::IntoComputedShape;
use tantale_core::utils::WrappedInfo;
use tantale_core::utils::record::{Recordable, Schema};
use tantale_core::{Domain, FidelityProblem, Problem, SolInfo};
use tantale_core::{
    FidOutcome, FuncState, HasFidelity, 
    HasStep, State, Sampler, Searchspace,
};

/// A helper macro to simplify the type signature of [`Hyperband`].
/// You can write:
/// ```ignore
/// let exp = load!(mono, hyperband!(asha!(RandomSearch)), Evaluated, (sp, cod), obj, (rec, check));
/// ```
/// or even:
/// ```ignore
/// let exp = load!(mono, hyperband!(asha!(tpe!(Univariate, UniformWeighter, LinearSplit))), Evaluated, (sp, cod), obj, (rec, check));
/// ```
#[macro_export]
macro_rules! hyperband {
    ($inner : ty) => {
        Hyperband<$inner, _, _, _>
    };
}

/// Metadata for Hyperband optimization steps, wrapping inner optimizer metadata.
///
/// Each evaluation iteration in Hyperband includes metadata about:
/// - Which bracket is currently executing
/// - The inner optimizer's own metadata (e.g., iteration number for SHA)
///
/// This allows tracking which bracket a solution belongs to when analyzing results.
#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
pub struct HyperbandInfo {
    /// The bracket index ($s$) for the current iteration.
    /// Ranges from $0$ (exploration-focused) to $s_{\max}$ (exploitation-focused).
    pub bracket: usize,
}
impl SolInfo for HyperbandInfo {}

impl Schema<()> for HyperbandInfo {
    type Parts = ();
    fn columns(_context: &()) -> Vec<String> {
        vec!["bracket".to_string()]
    }
}

impl Recordable<()> for HyperbandInfo {
    type Parts = ();
    fn record(&self, _context: &()) -> Vec<String> {
        vec![self.bracket.to_string()]
    }
}

impl HyperbandInfo {
    /// Creates a new instance of [`HyperbandInfo`] with the specified bracket index and inner information.
    ///
    /// # Arguments
    ///
    /// * `bracket` - The bracket index (0 to $s_{\max}$)
    /// * `inner_info` - The metadata produced by the inner optimizer for this iteration
    pub fn new(bracket: usize) -> Self {
        HyperbandInfo {
            bracket,
        }
    }
}

/// Hyperband multi-fidelity optimizer.
///
/// A wrapper optimizer that manages multiple brackets of an inner [`BudgetPruner`] (typically SHA or ASHA).
/// Hyperband automatically determines bracket configurations and executes them sequentially,
/// efficiently distributing a fixed budget across exploration and exploitation.
///
/// The information about maximum and minimum budget
/// are extracted from the inner optimizer.
///
/// # Workflow
///
/// ```text
///  Start Hyperband
///       |
///       v
///  +--------------------+
///  | Compute s_max      |  s_max = floor(log_eta(b_max/b_min))
///  +--------------------+
///       |
///       v
///  +--------------------+
///  | For s = s_max to 0 |  Loop through brackets
///  +--------------------+
///       |
///       v
///  +--------------------+
///  | Configure bracket: |
///  | b_min_s = b_min *  |
///  |     eta^(s_max-s)  |
///  | b_max_s = b_max    |
///  +--------------------+
///       |
///       v
///  +--------------------+
///  | Set inner optimizer|
///  | budgets            |
///  +--------------------+
///       |
///       v
///  +--------------------+
///  | Run inner          |
///  | optimizer          |  (SHA or ASHA)
///  | for bracket s      |
///  +--------------------+
///       |
///       v
///  +--------------------+
///  | Drain pending      |
///  | solutions from     |
///  | inner optimizer    |
///  +--------------------+
///        |
///        v
///       / \
///      /   \
///     /     \   No
///    /  All  \ ---> Move to next bracket
///    \ done? /         (loop back)
///     \     /
///      \   /
///       \ /
///        | Yes
///        v
///  +--------------------+
///  | Return best across |
///  | all brackets       |
///  +--------------------+
///
/// Bracket characteristics:
///   s=s_max: many configs, low fidelity   (exploration)
///   s=0:     few configs, high fidelity  (exploitation)
/// ```
///
/// # Type Parameters
///
/// - `Optim` - The inner optimizer (must implement both [`Optimizer`] and [`BudgetPruner`])
/// - `Out` - Output type satisfying [`FidOutcome`]
/// - `Scp` - Searchspace type
/// - `SInfo` - Solution info type for the inner optimizer
///
/// # See Also
///
/// - [`Sha`](crate::Sha) - Successive Halving (batch optimizer)
/// - [`Asha`](crate::Asha) - Asynchronous Successive Halving (sequential optimizer)
pub struct Hyperband<Prnr, P>
where
    Prnr: Pruner<P>,
    P: Problem,
{
    /// Minimum budget across all brackets ($b_{\min}$)
    pub budget_min: f64,
    /// Maximum budget across all brackets ($b_{\max}$)
    pub budget_max: f64,
    /// Scaling factor from the inner optimizer ($\eta$)
    pub scaling: f64,
    /// Maximum bracket index ($s_{\max} = \lfloor \log_\eta(b_{\max}/b_{\min}) \rfloor$)
    pub s_max: usize,
    /// Current bracket index being executed (ranges from 0 to `s_max`)
    pub current_s: usize,
    /// The inner optimizer state managing the current bracket
    pub inner: Prnr,
}

impl<Prnr, P> Hyperband<Prnr, P>
where
    Prnr: Pruner<P>,
    P: Problem,
{
    /// Creates a new [`Hyperband`] optimizer from an inner optimizer.
    ///
    /// This initializes the bracket configuration by extracting budget parameters from the
    /// inner optimizer. The brackets are computed as follows:
    /// - $s_{\max} = \lfloor \log_\eta(b_{\max}/b_{\min}) \rfloor$
    /// - For each bracket $s$: budget range is $[b_{\min} \times \eta^{s_{\max} - s}, b_{\max}]$
    ///
    /// # Arguments
    ///
    /// * `sampler` - An inner optimizer implementing both [`Optimizer`] and [`BudgetPruner`].
    ///   Typically [`Sha`](crate::Sha) or [`Asha`](crate::Asha).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sha = Sha::new(100, 1.0, 100.0, 2.0);
    /// let hyperband = Hyperband::new(sha);
    /// ```
    pub fn new(mut sampler: Prnr) -> Self {
        let (budget_min, budget_max) = sampler.get_budgets();
        let scaling = sampler.get_scaling();

        let s_max = (budget_max / budget_min).log(scaling).floor() as usize;
        let current_s = s_max; // To start with if current_s = 0, to correctly initialize

        let r = budget_max * scaling.powi(-(s_max as i32));
        sampler.set_budgets(budget_min, r);

        Hyperband {
            budget_min,
            budget_max,
            scaling,
            current_s,
            s_max,
            inner: sampler,
        }
    }
}

impl<Prnr, P, Opt> Sampler<P, Opt> for Hyperband<Prnr, P>
where
    Prnr: Pruner<P>,
    P: Problem,
    Opt: Domain,
{
    type Info = WrappedInfo<HyperbandInfo<>, Prnr::Info>;

    fn sample(&mut self,scp: &<P as Problem>::Scp,acc: &Accum<P>) -> Option<Shape<P>> {
        if self.inner.get_current_budget() < self.inner.get_budgets().1 {
            self.inner.sample(scp, acc)
        } else {
            let to_discard = self.inner.drain_one();

            if let Some(discard) = to_discard {
                to_discard
            } else {
                if self.0.current_s == 0 {
                    self.0.current_s = self.0.s_max;
                } else {
                    self.0.current_s -= 1;
                }
                let r = self.0.budget_max * self.0.scaling.powi(-(self.0.current_s as i32));
                self.inner.set_budgets(self.0.budget_min, r);
                self.inner.sample(scp, acc)
            }
        }
    }

    fn update(&mut self,x: &CompShape<P>,scp: &<P as Problem>::Scp,acc: &Accum<P>) {
        self.inner.update(x, scp, acc);
    }
}

//-------------//
//--- SERDE ---//
//-------------//

struct HyperbandStateVisitor<Prnr, P>(PhantomData<(Prnr, P)>)
where
    Prnr: Pruner<P>,
    P: Problem;
enum HBField {
    BudgetMin,
    BudgetMax,
    Scaling,
    SMax,
    CurrentS,
    Inner,
}

struct HBFieldVisitor;

impl<Prnr, P> Serialize for Hyperband<Prnr, P>
where
    Prnr: Pruner<P>,
    P: Problem,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("HyperbandState", 6)?;
        state.serialize_field("budget_min", &self.budget_min)?;
        state.serialize_field("budget_max", &self.budget_max)?;
        state.serialize_field("scaling", &self.scaling)?;
        state.serialize_field("s_max", &self.s_max)?;
        state.serialize_field("current_s", &self.current_s)?;
        state.serialize_field("inner", &self.inner.get_state())?;
        state.end()
    }
}

impl<Prnr, P> HyperbandStateVisitor<Prnr, P>
where
    Prnr: Pruner<P>,
    P: Problem,
{
    fn new() -> Self {
        HyperbandStateVisitor(PhantomData)
    }
}

impl<'de> Visitor<'de> for HBFieldVisitor {
    type Value = HBField;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter
            .write_str("`budget_min`, `budget_max`, `scaling`, `s_max`, `current_s` or `inner`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "budget_min" => Ok(HBField::BudgetMin),
            "budget_max" => Ok(HBField::BudgetMax),
            "scaling" => Ok(HBField::Scaling),
            "s_max" => Ok(HBField::SMax),
            "current_s" => Ok(HBField::CurrentS),
            "inner" => Ok(HBField::Inner),
            _ => Err(de::Error::unknown_field(
                value,
                &[
                    "budget_min",
                    "budget_max",
                    "scaling",
                    "s_max",
                    "current_s",
                    "inner",
                ],
            )),
        }
    }
}
impl<'de> Deserialize<'de> for HBField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(HBFieldVisitor)
    }
}

impl<'de, Prnr, P> Visitor<'de> for HyperbandStateVisitor<Prnr, P>
where
    Prnr: Pruner<P>,
    P: Problem,
{
    type Value = Hyperband<Prnr, P>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct HyperbandState")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let budget_min = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let budget_max = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(1, &self))?;
        let scaling = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(2, &self))?;
        let s_max = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(3, &self))?;
        let current_s = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(4, &self))?;
        let inner_state = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(5, &self))?;
        Ok(Hyperband {
            budget_min,
            budget_max,
            scaling,
            current_s,
            s_max,
            inner: Prnr::from_state(inner_state),
        })
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: de::MapAccess<'de>,
    {
        let mut budget_min = None;
        let mut budget_max = None;
        let mut scaling = None;
        let mut current_s = None;
        let mut s_max = None;
        let mut inner_state = None;

        while let Some(key) = map.next_key()? {
            match key {
                HBField::BudgetMin => {
                    if budget_min.is_some() {
                        return Err(de::Error::duplicate_field("budget_min"));
                    }
                    budget_min = Some(map.next_value()?);
                }
                HBField::BudgetMax => {
                    if budget_max.is_some() {
                        return Err(de::Error::duplicate_field("budget_max"));
                    }
                    budget_max = Some(map.next_value()?);
                }
                HBField::Scaling => {
                    if scaling.is_some() {
                        return Err(de::Error::duplicate_field("scaling"));
                    }
                    scaling = Some(map.next_value()?);
                }
                HBField::SMax => {
                    if s_max.is_some() {
                        return Err(de::Error::duplicate_field("s_max"));
                    }
                    s_max = Some(map.next_value()?);
                }
                HBField::CurrentS => {
                    if current_s.is_some() {
                        return Err(de::Error::duplicate_field("current_s"));
                    }
                    current_s = Some(map.next_value()?);
                }
                HBField::Inner => {
                    if inner_state.is_some() {
                        return Err(de::Error::duplicate_field("inner"));
                    }
                    inner_state = Some(map.next_value()?);
                }
            }
        }

        let budget_min = budget_min.ok_or_else(|| de::Error::missing_field("budget_min"))?;
        let budget_max = budget_max.ok_or_else(|| de::Error::missing_field("budget_max"))?;
        let scaling = scaling.ok_or_else(|| de::Error::missing_field("scaling"))?;
        let current_s = current_s.ok_or_else(|| de::Error::missing_field("current_s"))?;
        let s_max = s_max.ok_or_else(|| de::Error::missing_field("s_max"))?;
        let inner_state = inner_state.ok_or_else(|| de::Error::missing_field("inner"))?;

        Ok(Hyperband {
            budget_min,
            budget_max,
            scaling,
            current_s,
            s_max,
            inner: Prnr::from_state(inner_state),
        })
    }
}

impl<'de, Prnr, P> Deserialize<'de> for Hyperband<Prnr, P>
where
    Prnr: Pruner<P>,
    P: Problem,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(HyperbandStateVisitor::new())
    }
}
