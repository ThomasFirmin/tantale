use std::{fmt::Debug, marker::PhantomData};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::{self, Visitor}, ser::SerializeStruct};
use tantale_core::{Domain,  HasFidelity, HasState, HasStep, Problem, Sampled, Sampler, State, Step::{self, Partially},
aliases::{Accum, CompShape, Opt, Shape}, solution::shape::IntoComputedShape};
use tantale_macros::fidelity_bounds;

/// A helper macro to simplify the type signature of [`NoPruner`].
#[macro_export]
macro_rules! nopruner {
    ($sampler : ident) => {
        NoPruner<$sampler, _>
    };
    ($sampler : ty) => {
        NoPruner<$sampler, _>
    };
}

/// A pruner that does not prune any solution.
/// It stores a [`Partially`](tantale_core::Step::Partially) evaluated solution in the state, and returns it as is
/// until it becomes [`Evaluated`](tantale_core::Step::Evaluated).
/// 
/// It is used when the objective function is a [`MultiStep`](tantale_core::MultiStep),
/// but the [`Fidelity`](tantale_core::Fidelity) is ignored.
/// 
/// The objective is to evaluate a multi-step solution as if it was computed by a [`OneStep`](tantale_core::OneStep)
/// but with all the usual machinery between step.
/// 
/// An optional vector of [`Fidelity`](tantale_core::Fidelity) can be provided to store the fidelity of each step, but it is not used for pruning.
/// Be careful, the index of the fidelity vector must match the index of the [`Partially`](tantale_core::Step::Partially) step.
/// Index `0` corresponds to a [`Pending`](tantale_core::Step::Pending) step, index `1` to [`Partially(1)`](tantale_core::Step::Partially) step, and so on.
/// 
/// The inner [`Sampler`](tantale_core::Sampler) does not need to be [`Prunable`](tantale_core::Prunable).
/// The [`update`](tantale_core::Sampler::update) method of the sampler will only be called with fully evaluated solution.
#[fidelity_bounds(P)]
pub struct NoPruner<Smpl: Sampler<P, Opt<P>>, P: Problem>{
    pub sampler: Smpl,
    pub partially: Option<CompShape<P>>,
    pub fidelities: Option<Vec<f64>>,
}

#[fidelity_bounds(P)]
impl<Smpl: Sampler<P, Opt<P>>, P: Problem> Debug for NoPruner<Smpl, P> where P: Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoPruner")
        .field("sampler", &self.sampler)
        .field("partially", &self.partially)
        .field("fidelities", &self.fidelities).finish()
    }
}

#[fidelity_bounds(P)]
impl<Smpl: Sampler<P, Opt<P>>, P: Problem> Clone for NoPruner<Smpl, P> where P: Problem { 
    fn clone(&self) -> Self {
        NoPruner {
            sampler: self.sampler.clone(),
            partially: self.partially.clone(),
            fidelities: self.fidelities.clone(),
        }
    }

}

#[fidelity_bounds(P)]
impl<Smpl: Sampler<P, Opt<P>>, P: Problem> State for NoPruner<Smpl, P> where P: Problem { }


#[fidelity_bounds(P)]
impl<Smpl: Sampler<P, Opt<P>>, P: Problem> HasState for NoPruner<Smpl, P> {
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

#[fidelity_bounds(P)]
impl<Smpl: Sampler<P, Opt<P>>, P: Problem> NoPruner<Smpl, P> {
    /// Creates a new [`NoPruner`] with the given [`Sampler`](tantale_core::Sampler) and optional vector of [`f64`] for fidelities.
    pub fn new(sampler: Smpl, fidelities: Option<Vec<f64>>) -> Self {
        NoPruner {
            sampler,
            partially: None,
            fidelities,
        }
    }
}

#[fidelity_bounds(P, Opt)]
impl<Smpl: Sampler<P, Opt>, P: Problem, Opt:Domain> Sampler<P, Opt> for NoPruner<Smpl, P> 
{
    type Info = Smpl::Info;

    fn sample_info(&mut self, scp: &<P as Problem>::Scp, acc: &Accum<P>) -> Option<(Shape<P>, Sampled<Self::Info>)> {
        if let Some(partially) = self.partially.take() {
            let mut sol: Shape<P> = IntoComputedShape::extract(partially).0;
            if let Some(fids) = self.fidelities.as_ref() && let Partially(step) = sol.step(){
                sol.set_fidelity(fids[step as usize]);
            }
            Some((sol, Sampled::Reused))
        } else {
            let (mut shape, info) = self.sampler.sample_info(scp, acc)?;
            if let Some(fids) = self.fidelities.as_ref() {
                shape.set_fidelity(fids[0]);
            }
            Some((shape, info))
        }
    }

    fn update(&mut self,x: &CompShape<P>, _scp: &<P as Problem>::Scp, _acc: &Accum<P>) {
        match x.step() {
            Step::Partially(_) => self.partially = Some(x.clone()),
            Step::Evaluated => self.sampler.update(x, _scp, _acc),
            _ => (),
        }
    }
}
//-------------//
//--- SERDE ---//
//-------------//

struct NoPrunerStateVisitor<Smpl, P>(PhantomData<(Smpl, P)>)
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>>;

enum NoPrunerField {
    Sampler,
    Partially,
    Fidelities,
}

struct NoPrunerFieldVisitor;

#[fidelity_bounds(P)]
impl<Smpl, P> Serialize for NoPruner<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>>,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("NoPrunerState", 3)?;
        state.serialize_field("sampler", &self.sampler.get_state())?;
        state.serialize_field("partially", &self.partially)?;
        state.serialize_field("fidelities", &self.fidelities)?;
        state.end()
    }
}

impl<Smpl, P> NoPrunerStateVisitor<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>>,
{
    fn new() -> Self {
        NoPrunerStateVisitor(PhantomData)
    }
}

impl<'de> Visitor<'de> for NoPrunerFieldVisitor {
    type Value = NoPrunerField;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("`sampler`, `budgets`, `scaling`, `rung` or `current_budget`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "sampler" => Ok(NoPrunerField::Sampler),
            "partially" => Ok(NoPrunerField::Partially),
            "fidelities" => Ok(NoPrunerField::Fidelities),
            _ => Err(de::Error::unknown_field(
                value,
                &["sampler", "partially", "fidelities"],
            )),
        }
    }
}
impl<'de> Deserialize<'de> for NoPrunerField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(NoPrunerFieldVisitor)
    }
}

#[fidelity_bounds(P)]
impl<'de, Smpl, P> Visitor<'de> for NoPrunerStateVisitor<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>>,
{
    type Value = NoPruner<Smpl, P>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct NoPrunerState")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let sampler = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let partially = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(1, &self))?;
        let fidelities = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(2, &self))?;
        Ok(NoPruner {
            sampler: Smpl::from_state(sampler),
            partially,
            fidelities,
        })
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: de::MapAccess<'de>,
    {
        let mut sampler = None;
        let mut partially = None;
        let mut fidelities = None;

        while let Some(key) = map.next_key()? {
            match key {
                NoPrunerField::Sampler => {
                    if sampler.is_some() {
                        return Err(de::Error::duplicate_field("sampler"));
                    }
                    sampler = Some(map.next_value()?);
                }
                NoPrunerField::Partially => {
                    if partially.is_some() {
                        return Err(de::Error::duplicate_field("partially"));
                    }
                    partially = Some(map.next_value()?);
                }
                NoPrunerField::Fidelities => {
                    if fidelities.is_some() {
                        return Err(de::Error::duplicate_field("fidelities"));
                    }
                    fidelities = Some(map.next_value()?);
                }
            }
        }

        let sampler = sampler.ok_or_else(|| de::Error::missing_field("sampler"))?;
        let partially = partially.ok_or_else(|| de::Error::missing_field("partially"))?;
        let fidelities = fidelities.ok_or_else(|| de::Error::missing_field("fidelities"))?;

        Ok(NoPruner {
            sampler: Smpl::from_state(sampler),
            partially,
            fidelities,
        })
    }
}

#[fidelity_bounds(P)]
impl<'de, Smpl, P> Deserialize<'de> for NoPruner<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(NoPrunerStateVisitor::new())
    }
}
