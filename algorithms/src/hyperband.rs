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
use std::sync::Arc;

use serde::de::{self, Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};
use tantale_core::optimizer::opt::BudgetPruner;
use tantale_core::searchspace::CompShape;
use tantale_core::{Batch, BatchOptimizer, CSVWritable, OptInfo};
use tantale_core::{
    Codomain, Criteria, EmptyInfo, FidOutcome, FidelitySol, FuncState, HasFidelity, HasStep,
    IntoComputed, LinkOpt, OptState, Optimizer, RawObj, SId, Searchspace, SequentialOptimizer,
    SingleCodomain, SolutionShape, Stepped, searchspace::OptionCompShape,
};

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
pub struct HyperbandState<Optim, Out, Scp>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp, SInfo = EmptyInfo>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    pub budget_min: f64,
    pub budget_max: f64,
    pub scaling: f64,
    pub s_max: usize,
    pub current_s: usize,
    pub inner: Optim,
    _out: PhantomData<Out>,
    _scp: PhantomData<Scp>,
}

impl<Optim, Out, Scp> OptState for HyperbandState<Optim, Out, Scp>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp, SInfo = EmptyInfo>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
}

/// Information about the current state of the Hyperband optimization process.
#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
#[serde(bound(
    serialize = "Info: Serialize",
    deserialize = "Info: for<'a> Deserialize<'a>"
))]
pub struct HyperbandInfo<Info: OptInfo> {
    /// The current bracket index.
    pub bracket: usize,
    /// The inner information produced by the underlying optimizer for the current bracket.
    pub inner_info: Arc<Info>,
}
impl<Info: OptInfo> OptInfo for HyperbandInfo<Info> {}

impl<Info: OptInfo + CSVWritable<(), ()>> CSVWritable<(), ()> for HyperbandInfo<Info> {
    /// The header consists of the "bracket" column followed by the inner information columns.
    fn header(_elem: &()) -> Vec<String> {
        let head = Vec::from([String::from("bracket")]);
        let inner_head = Info::header(&());
        [head, inner_head].concat()
    }
    /// The "bracket" column contains the current bracket index, followed by the inner information values.
    fn write(&self, _comp: &()) -> Vec<String> {
        let head = Vec::from([self.bracket.to_string()]);
        let inner_head = self.inner_info.write(&());
        [head, inner_head].concat()
    }
}

impl<Info: OptInfo> HyperbandInfo<Info> {
    /// Creates a new instance of [`HyperbandInfo`] with the specified bracket index and inner information.
    pub fn new(bracket: usize, inner_info: Arc<Info>) -> Self {
        HyperbandInfo {
            bracket,
            inner_info,
        }
    }
}

pub struct Hyperband<Optim, Out, Scp>(pub HyperbandState<Optim, Out, Scp>)
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp, SInfo = EmptyInfo>
        + BudgetPruner<
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Scp::Opt,
            Out,
            Scp,
            SInfo = EmptyInfo,
        >,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>;

impl<Optim, Out, Scp> Hyperband<Optim, Out, Scp>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp, SInfo = EmptyInfo>
        + BudgetPruner<
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Scp::Opt,
            Out,
            Scp,
            SInfo = EmptyInfo,
        >,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    /// Creates a new instance of the Hyperband optimizer.
    /// This function initializes the internal state of the optimizer based on the provided sampler bounded by [`BudgetPruner`].
    /// The initial budget levels are computed based on the minimum and maximum budgets and the scaling factor from the sampler.
    /// The internal state is encapsulated in a `HyperbandState` struct, which includes all necessary information to resume optimization after checkpointing.
    pub fn new(sampler: Optim) -> Self {
        let (budget_min, budget_max) = sampler.get_budgets();
        let scaling = sampler.get_scaling();

        let s_max = (budget_max / budget_min).log(scaling).floor() as usize;
        let current_s = s_max;

        Hyperband(HyperbandState {
            budget_min,
            budget_max,
            scaling,
            current_s,
            s_max,
            inner: sampler,
            _out: PhantomData,
            _scp: PhantomData,
        })
    }
}

impl<Optim, Out, Scp> Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for Hyperband<Optim, Out, Scp>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp, SInfo = EmptyInfo>
        + BudgetPruner<
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Scp::Opt,
            Out,
            Scp,
            SInfo = EmptyInfo,
        >,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = HyperbandState<Optim, Out, Scp>;
    type Cod = Optim::Cod;
    type SInfo = Optim::SInfo;
    type Info = HyperbandInfo<Optim::Info>;

    /// Provides mutable access to the internal state of the [`Hyperband`] optimizer.
    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    /// Provides immutable access to the internal state of the [`Hyperband`] optimizer.
    fn get_state(&self) -> &Self::State {
        &self.0
    }

    /// Creates a new instance of the [`Hyperband`] optimizer from a given state.
    fn from_state(state: Self::State) -> Self {
        Hyperband(state)
    }
}

impl<Optim, Out, Scp, FnState>
    BatchOptimizer<
        FidelitySol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
    > for Hyperband<Optim, Out, Scp>
where
    Optim: BatchOptimizer<
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Scp::Opt,
            Out,
            Scp,
            Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
            SInfo = EmptyInfo,
        > + BudgetPruner<
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Scp::Opt,
            Out,
            Scp,
            SInfo = EmptyInfo,
        >,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<SId, Self::SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
{
    /// Computes the first batch of candidates for the Hyperband optimization process.
    /// It uses the inner [`BudgetPruner`] to generate the initial batch of candidates based on the current budget configuration.
    fn first_step(&mut self, scp: &Scp) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        let (pairs, info) = self.0.inner.first_step(scp).extract();
        let new_info = HyperbandInfo::new(self.0.current_s, info);
        Batch::new(pairs, new_info.into())
    }

    fn step(
        &mut self,
        x: Batch<
            SId,
            Self::SInfo,
            Self::Info,
            CompShape<Scp, FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Self::SInfo, Self::Cod, Out>,
        >,
        scp: &Scp,
    ) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        if self.0.inner.get_current_budget() < self.0.inner.get_budgets().1 {
            let (pairs, info) = x.extract();
            let extracted = Batch::new(pairs, info.inner_info.clone());

            let (pairs, info) = self.0.inner.step(extracted, scp).extract();
            let new_info = HyperbandInfo::new(self.0.current_s, info);
            Batch::new(pairs, new_info.into())
        } else {
            if self.0.current_s == 0 {
                self.0.current_s = self.0.s_max;
            } else {
                self.0.current_s -= 1;
            }

            let n = ((self.0.s_max as f64 + 1.) * self.0.scaling.powi(self.0.current_s as i32)
                / (self.0.current_s + 1) as f64)
                .ceil() as usize;
            self.0.inner.set_batch_size(n);
            let r = self.0.budget_max * self.0.scaling.powi(-(self.0.current_s as i32));
            self.0.inner.set_budgets(self.0.budget_min, r);

            let to_discard = self.0.inner.drain();

            if to_discard.is_empty() {
                self.first_step(scp)
            } else {
                let new_info = HyperbandInfo::new(self.0.current_s, Optim::Info::default().into());
                Batch::new(to_discard, new_info.into())
            }
        }
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.0.inner.set_batch_size(batch_size);
    }

    fn get_batch_size(&self) -> usize {
        self.0.inner.get_batch_size()
    }
}

impl<Optim, Out, Scp, FnState>
    SequentialOptimizer<
        FidelitySol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
    > for Hyperband<Optim, Out, Scp>
where
    Optim: SequentialOptimizer<
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Scp::Opt,
            Out,
            Scp,
            Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
            SInfo = EmptyInfo,
        > + BudgetPruner<
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Scp::Opt,
            Out,
            Scp,
            SInfo = EmptyInfo,
        >,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<SId, Self::SInfo> + HasStep + HasFidelity,
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
    ) -> Scp::SolShape {
        if self.0.inner.get_current_budget() < self.0.inner.get_budgets().1 {
            self.0.inner.step(x, scp)
        } else {
            let to_discard = x.map_or(self.0.inner.drain_one(), |mut comp| {
                comp.discard();
                Some(IntoComputed::extract(comp).0)
            });

            if let Some(discard) = to_discard {
                discard
            } else {
                if self.0.current_s == 0 {
                    self.0.current_s = self.0.s_max;
                } else {
                    self.0.current_s -= 1;
                }
                let r = self.0.budget_max * self.0.scaling.powi(-(self.0.current_s as i32));
                self.0.inner.set_budgets(self.0.budget_min, r);
                self.0.inner.step(None, scp)
            }
        }
    }
}

//-------------//
//--- SERDE ---//
//-------------//

struct HyperbandStateVisitor<Optim, Out, Scp>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp, SInfo = EmptyInfo>,
    Optim::State: Serialize + for<'de> Deserialize<'de>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    _optim: PhantomData<Optim>,
    _out: PhantomData<Out>,
    _scp: PhantomData<Scp>,
}

enum HBField {
    BudgetMin,
    BudgetMax,
    Scaling,
    SMax,
    CurrentS,
    Inner,
}

struct HBFieldVisitor;

impl<Optim, Out, Scp> Serialize for HyperbandState<Optim, Out, Scp>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp, SInfo = EmptyInfo>,
    Optim::State: Serialize,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
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

impl<Optim, Out, Scp> HyperbandStateVisitor<Optim, Out, Scp>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp, SInfo = EmptyInfo>,
    Optim::State: Serialize + for<'de> Deserialize<'de>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    fn new() -> Self {
        HyperbandStateVisitor {
            _optim: PhantomData,
            _out: PhantomData,
            _scp: PhantomData,
        }
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

impl<'de, Optim, Out, Scp> Visitor<'de> for HyperbandStateVisitor<Optim, Out, Scp>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp, SInfo = EmptyInfo>,
    Optim::State: Serialize + Deserialize<'de>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type Value = HyperbandState<Optim, Out, Scp>;

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
        let current_s = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(3, &self))?;
        let s_max = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(4, &self))?;
        let inner_state = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(5, &self))?;
        Ok(HyperbandState {
            budget_min,
            budget_max,
            scaling,
            current_s,
            s_max,
            inner: Optim::from_state(inner_state),
            _out: PhantomData,
            _scp: PhantomData,
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

        Ok(HyperbandState {
            budget_min,
            budget_max,
            scaling,
            current_s,
            s_max,
            inner: Optim::from_state(inner_state),
            _out: PhantomData,
            _scp: PhantomData,
        })
    }
}

impl<'de, Optim, Out, Scp> Deserialize<'de> for HyperbandState<Optim, Out, Scp>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp, SInfo = EmptyInfo>,
    Optim::State: Serialize + Deserialize<'de>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(HyperbandStateVisitor::new())
    }
}
