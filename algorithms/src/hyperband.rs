//! # Hyperband
//!
//! This module implements the batch, SHA-based Hyperband variant. It divides
//! the fidelity range into bracket configurations and runs those brackets
//! through an inner [`Sha`] optimizer. The brackets vary the
//! starting fidelity and initial batch size to balance broad low-fidelity
//! exploration with smaller high-fidelity searches.
//!
//! Create it from SHA with [`Sha::into_hyperband`](crate::Sha::into_hyperband).
//! That method derives the bracket count and initial configuration from the
//! SHA budget range and scaling factor. The public [`Hyperband::new`] constructor
//! accepts those derived values explicitly and should generally not be needed.
//!
//! ASHA and MO-ASHA provide their own `into_hyperband` conversions to
//! `BrackettedAsha` and `BrackettedMoAsha`. Those are distinct bracketed
//! optimizers; this module's [`Hyperband`] contains SHA.
//!
//! # Pseudo-code
//!
//! **Hyperband**
//! ---
//! **Inputs**
//! 1. &emsp; $\mathcal{O}$ &emsp;&emsp; *An inner optimizer implementing [`Pruner`] (e.g., SHA or ASHA)*
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
//! # Reference
//!
//! Li et al., [*Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*](https://arxiv.org/abs/1603.06560).

use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use serde::de::{self, Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};
use tantale_core::aliases::{Accum, CompShape, Opt, Shape};
use tantale_core::optimizer::opt::{Batched, Pruner};
use tantale_core::utils::WrappedInfo;
use tantale_core::{Domain, Problem};
use tantale_core::{HasState, Orderable, Prunable, Sampled};
use tantale_core::{Sampler, State};
use tantale_macros::fidelity_bounds;

use crate::Sha;
use crate::utils::BracketInfo;

/// A helper macro for the `Hyperband<InnerSampler, Problem>` type.
/// 
/// # Example
/// ```ignore
/// hyperband!(RandomSearch) // expands to Hyperband<RandomSearch, _>
/// ```
#[macro_export]
macro_rules! hyperband {
    ($inner : ty) => {
        Hyperband<$inner, _>
    };
}

/// SHA-based Hyperband optimizer for multi-fidelity problems.
///
/// The inner SHA optimizer is reconfigured for successive brackets. Prefer
/// [`Sha::into_hyperband`](crate::Sha::into_hyperband), which calculates the
/// bracket parameters from the SHA budgets and scaling factor.
#[fidelity_bounds(P)]
pub struct Hyperband<Smpl, P>
where
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
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
    pub inner: Sha<Smpl, P>,
    _p: PhantomData<P>,
}

#[fidelity_bounds(P)]
impl<Smpl, P> Hyperband<Smpl, P>
where
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    pub fn new(
        budget_min: f64,
        budget_max: f64,
        scaling: f64,
        s_max: usize,
        inner: Sha<Smpl, P>,
    ) -> Self {
        let current_s = s_max; // Start with the highest bracket
        Hyperband {
            budget_min,
            budget_max,
            scaling,
            s_max,
            current_s,
            inner,
            _p: PhantomData,
        }
    }
}

#[fidelity_bounds(P)]
impl<Smpl, P> Clone for Hyperband<Smpl, P>
where
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn clone(&self) -> Self {
        Self {
            budget_min: self.budget_min,
            budget_max: self.budget_max,
            scaling: self.scaling,
            s_max: self.s_max,
            current_s: self.current_s,
            inner: self.inner.clone(),
            _p: self._p,
        }
    }
}

#[fidelity_bounds(P)]
impl<Smpl, P> Debug for Hyperband<Smpl, P>
where
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hyperband")
            .field("budget_min", &self.budget_min)
            .field("budget_max", &self.budget_max)
            .field("scaling", &self.scaling)
            .field("s_max", &self.s_max)
            .field("current_s", &self.current_s)
            .field("inner", &self.inner)
            .finish()
    }
}

#[fidelity_bounds(P)]
impl<Smpl, P> Hyperband<Smpl, P>
where
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    #[allow(clippy::type_complexity)]
    fn wrap(
        &self,
        sampled: Option<(Shape<P>, Sampled<Smpl::Info>)>,
    ) -> Option<(Shape<P>, Sampled<WrappedInfo<Smpl::Info, BracketInfo>>)> {
        sampled.map(|(sol, info)| match info {
            Sampled::Fresh(inner) => {
                let hinfo = Arc::new(BracketInfo(self.current_s));
                (
                    sol,
                    Sampled::Fresh(Arc::new(WrappedInfo::new(inner, hinfo))),
                )
            }
            Sampled::Reused => (sol, Sampled::Reused),
        })
    }
}

#[fidelity_bounds(P)]
impl<Smpl, P> State for Hyperband<Smpl, P>
where
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
}

#[fidelity_bounds(P)]
impl<Smpl, P> HasState for Hyperband<Smpl, P>
where
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
impl<Smpl, P, Opt> Sampler<P, Opt> for Hyperband<Smpl, P>
where
    Smpl: Sampler<P, Opt> + Prunable,
    CompShape<P>: Orderable,
    Opt: Domain,
{
    type Info = WrappedInfo<Smpl::Info, BracketInfo>;

    fn sample_info(
        &mut self,
        scp: &<P as Problem>::Scp,
        acc: &Accum<P>,
    ) -> Option<(Shape<P>, Sampled<Self::Info>)> {
        if !self.inner.bracket_complete() {
            let sampled = self.inner.sample_info(scp, acc);
            self.wrap(sampled)
        } else {
            if self.current_s == 0 {
                self.current_s = self.s_max;
            } else {
                self.current_s -= 1;
            }
            let n = (self.s_max as f64 + 1.) / (self.current_s as f64 + 1.)
                * self.scaling.powi(self.current_s as i32);
            self.inner.set_batch_size((n.ceil() as usize).max(1));
            let r = self.budget_max * self.scaling.powi(-(self.current_s as i32));
            self.inner.set_budgets(r, self.budget_max);
            let sampled = self.inner.sample_info(scp, acc);
            self.wrap(sampled)
        }
    }

    fn update(&mut self, x: &CompShape<P>, scp: &<P as Problem>::Scp, acc: &Accum<P>) {
        self.inner.update(x, scp, acc);
    }
}

//-------------//
//--- SERDE ---//
//-------------//

#[fidelity_bounds(P)]
struct HyperbandStateVisitor<Smpl, P>(PhantomData<(Smpl, P)>)
where
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable;

enum HBField {
    BudgetMin,
    BudgetMax,
    Scaling,
    SMax,
    CurrentS,
    Inner,
}

struct HBFieldVisitor;

#[fidelity_bounds(P)]
impl<Smpl, P> Serialize for Hyperband<Smpl, P>
where
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
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

#[fidelity_bounds(P)]
impl<Smpl, P> HyperbandStateVisitor<Smpl, P>
where
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn new() -> Self {
        HyperbandStateVisitor(PhantomData)
    }
}

impl<'de> Visitor<'de> for HBFieldVisitor {
    type Value = HBField;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("Hyperband state field")
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

#[fidelity_bounds(P)]
impl<'de, Smpl, P> Visitor<'de> for HyperbandStateVisitor<Smpl, P>
where
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    type Value = Hyperband<Smpl, P>;

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
            .ok_or_else(|| de::Error::invalid_length(8, &self))?;
        Ok(Hyperband {
            budget_min,
            budget_max,
            scaling,
            current_s,
            s_max,
            inner: Sha::<Smpl, P>::from_state(inner_state),
            _p: PhantomData,
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
            inner: Sha::<Smpl, P>::from_state(inner_state),
            _p: PhantomData,
        })
    }
}

#[fidelity_bounds(P)]
impl<'de, Smpl, P> Deserialize<'de> for Hyperband<Smpl, P>
where
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(HyperbandStateVisitor::new())
    }
}
