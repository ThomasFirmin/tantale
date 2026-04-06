use crate::{
    GridDom,
    domain::{
        Domain, PreDomain, TypeDom,
        bool::Bool,
        bounded::{Bounded, BoundedBounds, RangeDomain},
        grid::GridBounds,
        mixed::{Mixed, MixedTypeDom},
        onto::{Onto, OntoDom},
    },
    errors::OntoError,
    recorder::csv::CSVWritable,
    sampler::{BoundedDistribution, Sampler},
};

use num::cast::AsPrimitive;
use rand::prelude::Rng;
use serde::{Deserialize, Serialize};
use std::{fmt, ops::RangeInclusive};

/// A [`f64`] [`Domain`] domain within `[0.0,1.0]`.
///
///  # Examples
///
/// ```
/// use tantale::core::{Unit, Domain,Uniform};
/// let dom : Unit = Unit::new(Uniform);
///
/// let mut rng = rand::rng();
/// let sample = dom.sample(&mut rng);
/// assert!(dom.is_in(&sample));
/// assert_eq!(*dom.bounds.start(), 0.0);
/// assert_eq!(*dom.bounds.end(), 1.0);
/// assert_eq!(dom.mid, 0.5);
/// assert_eq!(dom.width, 1.0);
/// ```
pub struct Unit {
    pub bounds: RangeInclusive<f64>,
    pub mid: f64,
    pub width: f64,
    pub sampler: BoundedDistribution,
}

impl Unit {
    /// Fabric for a [`Unit`] [`Domain`].
    pub fn new<S: Sampler<Self> + Into<BoundedDistribution>>(sampler: S) -> Unit {
        Unit {
            bounds: RangeInclusive::new(0.0, 1.0),
            mid: 0.5,
            width: 1.0,
            sampler: sampler.into(),
        }
    }
}

impl PartialEq for Unit {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl PreDomain for Unit {}
impl Domain for Unit {
    type TypeDom = f64;

    /// Sample a `f64` using the inner [`BoundedDistribution`] of [`Unit`].
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::TypeDom {
        self.sampler.sample(self, rng)
    }

    fn is_in(&self, item: &Self::TypeDom) -> bool {
        self.bounds.contains(item)
    }
}

impl RangeDomain for Unit {
    fn get_bounds(&self) -> RangeInclusive<Self::TypeDom> {
        self.bounds.clone()
    }
}

impl std::clone::Clone for Unit {
    fn clone(&self) -> Self {
        Unit::new(self.sampler)
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}

impl fmt::Debug for Unit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}

impl<Out> Onto<Bounded<Out>> for Unit
where
    Out: BoundedBounds + Serialize + for<'a> Deserialize<'a>,
    f64: AsPrimitive<Out>,
{
    type Item = TypeDom<Unit>;
    type TargetItem = TypeDom<Bounded<Out>>;
    /// [`Onto`] function between a [`Unit`] [`Domain`] and a [`Bounded`] [`Domain`].
    ///
    /// Considering $l_{out}$, $u_{out}$ the lower and upper bounds of
    /// the output [`Bounded`] [`Domain`], and $x$ the item to be mapped,
    /// the mapping is given by
    ///
    /// $$ x \times (u_{out}-l_{out}) + l_{out}$$
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Unit`]`<In>`.
    /// * `target` - A borrowed targetted [`Bounded`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Unit`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Bounded`] domain.
    fn onto(
        &self,
        item: &Self::Item,
        target: &Bounded<Out>,
    ) -> Result<Self::TargetItem, OntoError> {
        if self.is_in(item) {
            let a: f64 = *item;
            let c: f64 = target.width.as_();
            let mapped: Out = (a * c).as_() + *target.bounds.start();

            if target.is_in(&mapped) {
                Ok(mapped)
            } else {
                Err(OntoError(format!("{} input not in {}", item, self)))
            }
        } else {
            Err(OntoError(format!("{} input not in {}", item, self)))
        }
    }
}
impl<Out> OntoDom<Bounded<Out>> for Unit
where
    Out: BoundedBounds + Serialize + for<'a> Deserialize<'a>,
    f64: AsPrimitive<Out>,
{
}

impl Onto<Bool> for Unit
where
    f64: AsPrimitive<f64>,
{
    type Item = TypeDom<Unit>;
    type TargetItem = TypeDom<Bool>;
    /// [`Onto`] function between a [`Unit`] [`Domain`] and a [`Bool`][`Domain`].
    ///
    /// Considering $x$ the `item`, returns `true` if $x>0.5$
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Unit`]`<In>`.
    /// * `target` - A borrowed targetted [`Bool`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Unit`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Bool`] domain.
    fn onto(&self, item: &Self::Item, _target: &Bool) -> Result<Self::TargetItem, OntoError> {
        if self.is_in(item) {
            Ok(*item > 0.5)
        } else {
            Err(OntoError(format!("{} input not in {}", item, self)))
        }
    }
}
impl OntoDom<Bool> for Unit {}

impl<Out: GridBounds> Onto<GridDom<Out>> for Unit {
    type Item = TypeDom<Unit>;
    type TargetItem = Out;
    /// [`Onto`] function between a [`Unit`] [`Domain`] and a [`GridDom`][`Domain`].
    ///
    /// Considering $\ell_{en}$ the length of `values` of the [`GridDom`] [`Domain`].
    /// The variable $x$ is the item to be mapped.
    /// The mapping is given by mapping item to an index of `values` in [`GridDom`]:
    ///
    /// $$ i = \\left\\lfloor x \times \ell_{en} \\right\\rfloor $$
    /// $$ \\texttt{index} = \\begin{cases}
    ///    i & \\text{if } i < \ell_{en} \\\\
    ///    i -1 & \\text{if } i = \ell_{en}
    /// \\end{cases} $$
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Unit`]`<In>`.
    /// * `target` - A borrowed targetted [`GridDom`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Unit`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`GridDom`] domain.
    fn onto(
        &self,
        item: &Self::Item,
        target: &GridDom<Out>,
    ) -> Result<Self::TargetItem, OntoError> {
        if self.is_in(item) {
            let a: f64 = item.as_();
            let c: f64 = target.values.len().as_();
            let idx = (a * c) as usize;
            let idx = if idx == target.values.len() {
                idx - 1
            } else {
                idx
            };
            let mapped = target.values[idx].clone();

            if target.is_in(&mapped) {
                Ok(mapped)
            } else {
                Err(OntoError(format!("{} input not in {}", item, self)))
            }
        } else {
            Err(OntoError(format!("{} input not in {}", item, self)))
        }
    }
}
impl<Out: GridBounds> OntoDom<GridDom<Out>> for Unit {}

impl Onto<Mixed> for Unit {
    type Item = TypeDom<Unit>;
    type TargetItem = TypeDom<Mixed>;
    /// [`Onto`] function between a [`Unit`] [`Domain`] and a [`Mixed`][`Domain`].
    ///
    /// Considering $x$ the `item`, the mapping is done depending on the target [`Mixed`] [`Domain`] variant
    /// using the inner `d` [`Domain`] using [`Onto::onto`]
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Unit`]`<In>`.
    /// * `target` - A borrowed targetted [`Mixed`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Unit`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Mixed`] domain.
    fn onto(&self, item: &Self::Item, target: &Mixed) -> Result<Self::TargetItem, OntoError> {
        match target {
            Mixed::Real(target) => self.onto(item, target).map(MixedTypeDom::Real),
            Mixed::Nat(target) => self.onto(item, target).map(MixedTypeDom::Nat),
            Mixed::Int(target) => self.onto(item, target).map(MixedTypeDom::Int),
            Mixed::Bool(target) => self.onto(item, target).map(MixedTypeDom::Bool),
            Mixed::Cat(target) => self.onto(item, target).map(MixedTypeDom::Cat),
            Mixed::GridReal(target) => self.onto(item, target).map(MixedTypeDom::GridReal),
            Mixed::GridNat(target) => self.onto(item, target).map(MixedTypeDom::GridNat),
            Mixed::GridInt(target) => self.onto(item, target).map(MixedTypeDom::GridInt),
            _ => Err(OntoError(format!(
                "Converting the value {:?} from {:?} onto Mixed is not implemented, and it should not occur.",
                item, self
            ))),
        }
    }
}
impl OntoDom<Mixed> for Unit {}

impl Onto<Unit> for Unit
where
    f64: AsPrimitive<f64>,
{
    type Item = TypeDom<Unit>;
    type TargetItem = TypeDom<Unit>;
    /// [`Onto`] function between a [`Unit`] [`Domain`] and a [`Unit`][`Domain`].
    ///
    /// Considering $x$ the `item`, returns a copy of $x$.
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Unit`]`<In>`.
    /// * `target` - A borrowed targetted [`Unit`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Unit`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Unit`] domain.
    fn onto(&self, item: &Self::Item, _target: &Unit) -> Result<Self::TargetItem, OntoError> {
        Ok(*item)
    }
}
impl OntoDom<Unit> for Unit {}

impl From<Mixed> for Unit {
    fn from(value: Mixed) -> Self {
        match value {
            Mixed::Unit(d) => d,
            _ => unreachable!("Can only From<BaseDom> with Unit."),
        }
    }
}

impl CSVWritable<(), f64> for Unit {
    fn header(_elem: &()) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, comp: &f64) -> Vec<String> {
        Vec::from([comp.to_string()])
    }
}
