//! A [`Bounded`]`<T>` domain defines any domain that can be expressed
//! by lower and upper bounds. The main struct is [`Bounded`], defined
//! by a [`RangeInclusive`](std::ops::RangeInclusive). The trait
//! [`DomainBounded`] allows a more general definition.
//!
//! There are 3 type alises :
//! * [`Real`] for [`Bounded`]`<f64>`
//!     * For example the learning rate of the gradient descent algorithm, $[0.001, 0.1]$.
//! * [`Nat`] for [`Bounded`]`<u64>`
//!     * For example the number of neurons within a layer, $[10, 1000]$.
//! * [`Int`] for [`Bounded`]`<i64>`
//!     * Integer hyperparameters (that can be negative) are less common in machine learning.
//!     * For example, [padding](https://docs.jax.dev/en/latest/_autosummary/jax.lax.pad.html) in [Jax](https://docs.jax.dev/en/latest), $[-5,5]$.
//!
//! # Examples
//!
//! ```
//! use tantale::core::{Bounded, Domain, Uniform};
//! let dom : Bounded<u8> = Bounded::new(0, 255, Uniform);
//!
//! let mut rng = rand::rng();
//! let sample = dom.sample(&mut rng);
//! assert!(dom.is_in(&sample));
//! assert_eq!(*dom.bounds.start(), 0);
//! assert_eq!(*dom.bounds.end(), 255);
//! assert_eq!(dom.mid, 127);
//! assert_eq!(dom.width, 255);
//! ```

use crate::{
    GridDomDistribution,
    domain::{
        Domain, PreDomain, TypeDom,
        bool::Bool,
        grid::{GridBounds, GridDom},
        mixed::{Mixed, MixedTypeDom},
        onto::{Onto, OntoDom},
        unit::Unit,
    },
    errors::OntoError,
    recorder::csv::CSVWritable,
    sampler::{BoundedDistribution, Sampler},
};

use num::{Num, NumCast, cast::AsPrimitive};
use rand::{Rng, distr::uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{self, Debug, Display},
    ops::RangeInclusive,
};

// _-_-_-_-_-_-__-_-_-_-_-_-_-_
// Bounded domain

pub trait RangeDomain: Domain
where
    Self::TypeDom: BoundedBounds,
{
    fn get_bounds(&self) -> RangeInclusive<Self::TypeDom>;
}

/// A shortcut for the bounds of the generic type `<T>` in [`Bounded`]`<T>`
pub trait BoundedBounds:
    Num
    + NumCast
    + PartialEq
    + PartialOrd
    + Copy
    + Clone
    + SampleUniform
    + AsPrimitive<f64>
    + Display
    + Debug
    + Default
    + Serialize
    + for<'a> Deserialize<'a>
{
}
impl<T> BoundedBounds for T where
    T: Num
        + NumCast
        + PartialEq
        + PartialOrd
        + Copy
        + Clone
        + SampleUniform
        + AsPrimitive<f64>
        + Display
        + Debug
        + Default
        + Serialize
        + for<'a> Deserialize<'a>
{
}

/// A generic [`Bounded`] [`Domain`] with a numerical `lower` and `upper` bounds, modeled
/// by a [`RangeInclusive`].
///
/// # Attributes
/// * `bounds` - A [`RangeInclusive`] object of type `<T>`.
/// * `mid` - Middle point of the [`Bounded`] [`Domain`]. $\frac{\texttt{lower}+\texttt{upper}}{2}$
/// * `width` - Width of the [`Bounded`] [`Domain`]. $\texttt{upper}-\texttt{lower}$
///
pub struct Bounded<T: BoundedBounds> {
    pub bounds: RangeInclusive<T>,
    pub mid: T,
    pub width: T,
    pub sampler: BoundedDistribution,
}

impl<T: BoundedBounds> Bounded<T> {
    /// Fabric for a [`Bounded`].
    ///
    /// * The `mid` attribute is automatically computed with $\frac{\texttt{lower}+\texttt{upper}}{2}$.
    /// * The `width` attribute is automatically computed with $\texttt{upper}-\texttt{lower}$.
    ///
    /// # Parameters
    /// * `lower` - Lower bound of the [`Bounded`] [`Domain`].
    /// * `upper` - Upper bound of the [`Bounded`] [`Domain`].
    ///
    pub fn new<S: Sampler<Self> + Into<BoundedDistribution>>(
        lower: T,
        upper: T,
        sampler: S,
    ) -> Bounded<T> {
        if lower < upper {
            let mid = (upper + lower) / T::from(2).unwrap();
            let width = upper - lower;
            Bounded {
                bounds: std::ops::RangeInclusive::new(lower, upper),
                mid,
                width,
                sampler: sampler.into(),
            }
        } else {
            panic!("Boundaries error, {} is not < {}.", lower, upper);
        }
    }

    /// Fabric for a [`Bounded`] dedicated to be wrapped in a [`Grid`](crate::domain::grid::Grid).
    /// Generates a [`GridDom`] containing discretized `values` of type `T`.
    ///
    /// # Parameters
    /// * `values` - A list of values that will be used as bounds of the [`Bounded`] domain.
    pub fn grid<
        I: IntoIterator<Item = Item>,
        Item: Into<T>,
        S: Sampler<GridDom<T>> + Into<GridDomDistribution>,
    >(
        values: I,
        sampler: S,
    ) -> GridDom<T> {
        GridDom::new(values, sampler)
    }
}

impl<T: BoundedBounds> PartialEq for Bounded<T> {
    fn eq(&self, other: &Self) -> bool {
        self.bounds == other.bounds
    }
}

impl<T: BoundedBounds> PreDomain for Bounded<T> {}
impl<T: BoundedBounds> Domain for Bounded<T> {
    type TypeDom = T;

    /// Sample a `T` using the inner [`BoundedDistribution`] of [`Bounded`].
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::TypeDom {
        self.sampler.sample(self, rng)
    }

    /// Method to check if a given point is in the domain.
    fn is_in(&self, item: &T) -> bool {
        self.bounds.contains(item)
    }
}

impl<T: BoundedBounds> RangeDomain for Bounded<T> {
    fn get_bounds(&self) -> RangeInclusive<Self::TypeDom> {
        self.bounds.clone()
    }
}

impl<T: BoundedBounds> std::clone::Clone for Bounded<T> {
    fn clone(&self) -> Self {
        Bounded::new(*self.bounds.start(), *self.bounds.end(), self.sampler)
    }
}

impl<T: BoundedBounds> fmt::Display for Bounded<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}

impl<T: BoundedBounds> fmt::Debug for Bounded<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}

/// [`Onto`] function between [`Bounded`] [`Domain`].
///
/// Considering $l_{in}$, $u_{in}$, $l_{out}$ and $u_{out}$ the lower and upper bounds of
/// the input [`Bounded`] [`Domain`] and the output [`Bounded`] [`Domain`], and $x$ the item to be mapped,
/// the mapping is given by
///
/// $$ \frac{x-l_{in}}{u_{in}-l_{in}} \times (u_{out}-l_{out}) + l_{out}$$
impl<In, Out> Onto<Bounded<Out>> for Bounded<In>
where
    In: BoundedBounds,
    Out: BoundedBounds,
    f64: AsPrimitive<Out>,
{
    type Item = TypeDom<Bounded<In>>;
    type TargetItem = TypeDom<Bounded<Out>>;
    /// [`Onto`] function between a [`Bounded`] and another [`Bounded`] [`Domain`].
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Bounded`]`<In>`.
    /// * `target` - A borrowed targetted [`Bounded`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Bounded`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Bounded`] domain.
    fn onto(
        &self,
        item: &Self::Item,
        target: &Bounded<Out>,
    ) -> Result<Self::TargetItem, OntoError> {
        if self.is_in(item) {
            let a: f64 = (*item - *self.bounds.start()).as_();
            let b: f64 = self.width.as_();
            let c: f64 = target.width.as_();
            let mapped: Out = (a / b * c).as_() + *target.bounds.start();

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
impl<In, Out> OntoDom<Bounded<Out>> for Bounded<In>
where
    In: BoundedBounds,
    Out: BoundedBounds,
    f64: AsPrimitive<Out>,
{
}

impl<In> Onto<Bool> for Bounded<In>
where
    In: BoundedBounds,
    f64: AsPrimitive<In>,
{
    type Item = TypeDom<Bounded<In>>;
    type TargetItem = TypeDom<Bool>;
    /// [`Onto`] function between a [`Bounded`] [`Domain`] and a [`Bool`][`Domain`].
    ///
    /// Considering $l_{in}$ and $u_{in}$ the lower and upper bounds of
    /// the input [`Bounded`] [`Domain`], and $x$ the `item`, returns `true` if $x>\frac{u_{in}-l_{in}}{2}$
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Bounded`]`<In>`.
    /// * `target` - A borrowed targetted [`Bool`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Bounded`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Bool`] domain.
    fn onto(&self, item: &Self::Item, _target: &Bool) -> Result<Self::TargetItem, OntoError> {
        if self.is_in(item) {
            Ok(*item > self.mid)
        } else {
            Err(OntoError(format!("{} input not in {}", item, self)))
        }
    }
}
impl<In> OntoDom<Bool> for Bounded<In>
where
    In: BoundedBounds,
    f64: AsPrimitive<In>,
{
}

impl<In> Onto<Unit> for Bounded<In>
where
    In: BoundedBounds,
{
    type Item = TypeDom<Bounded<In>>;
    type TargetItem = TypeDom<Unit>;
    /// [`Onto`] function between a [`Bounded`] [`Domain`] and a [`Unit`][`Domain`].
    ///
    /// Considering $l_{in}$, $u_{in}$ the lower and upper bounds of
    /// the input [`Bounded`] [`Domain`], and $x$ the item to be mapped,
    /// the mapping is given by
    ///
    /// $$ \frac{x-l_{in}}{u_{in}-l_{in}}$$
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Bounded`]`<In>`.
    /// * `target` - A borrowed targetted [`Unit`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Bounded`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Unit`] domain.
    ///
    fn onto(&self, item: &Self::Item, target: &Unit) -> Result<Self::TargetItem, OntoError> {
        if self.is_in(item) {
            let a: f64 = (*item - *self.bounds.start()).as_();
            let b: f64 = self.width.as_();
            let mapped: f64 = a / b;

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
impl<In> OntoDom<Unit> for Bounded<In>
where
    In: BoundedBounds,
    f64: AsPrimitive<In>,
{
}

impl<In, Out> Onto<GridDom<Out>> for Bounded<In>
where
    In: BoundedBounds,
    Out: GridBounds,
    f64: AsPrimitive<In>,
{
    type Item = TypeDom<Bounded<In>>;
    type TargetItem = TypeDom<GridDom<Out>>;
    /// [`Onto`] function between a [`Bounded`] [`Domain`] and a [`Bounded`][`Domain`].
    ///
    /// Considering $l_{in}$, $u_{in}$ the lower and upper bounds of
    /// the input [`Bounded`] [`Domain`] and $\ell_{en}$ the length of `values` of the [`Bounded`] [`Domain`].
    /// The variable $x$ is the item to be mapped.
    /// The mapping is given by mapping item to an index of `values` in [`Bounded`]:
    ///
    /// $$ i = \\left\\lfloor \frac{x-l_{in}}{u_{in}-l_{in}} \times \ell_{en} \\right\\rfloor $$
    /// $$ \\texttt{index} = \\begin{cases}
    ///    i & \\text{if } i < \ell_{en} \\\\
    ///    i -1 & \\text{if } i = \ell_{en}
    /// \\end{cases} $$
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Bounded`]`<In>`.
    /// * `target` - A borrowed targetted [`Bounded`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Bounded`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Bounded`] domain.
    fn onto(
        &self,
        item: &Self::Item,
        target: &GridDom<Out>,
    ) -> Result<Self::TargetItem, OntoError> {
        if self.is_in(item) {
            let a: f64 = (*item - *self.bounds.start()).as_();
            let b: f64 = self.width.as_();
            let c: f64 = target.values.len().as_();
            let idx = (a / b * c) as usize;
            let idx = if idx == target.values.len() {
                idx - 1
            } else {
                idx
            };
            Ok(target.values[idx].clone())
        } else {
            Err(OntoError(format!("{} input not in {}", item, self)))
        }
    }
}
impl<In, Out> OntoDom<GridDom<Out>> for Bounded<In>
where
    In: BoundedBounds,
    Out: GridBounds,
    f64: AsPrimitive<In>,
{
}

impl<In> Onto<Mixed> for Bounded<In>
where
    In: BoundedBounds,
    f64: AsPrimitive<In>,
{
    type Item = TypeDom<Bounded<In>>;
    type TargetItem = TypeDom<Mixed>;
    /// [`Onto`] function between a [`Bounded`] [`Domain`] and a [`Mixed`][`Domain`].
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Bounded`]`<In>`.
    /// * `target` - A borrowed targetted [`Mixed`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Bounded`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Mixed`] domain.
    fn onto(&self, item: &Self::Item, target: &Mixed) -> Result<Self::TargetItem, OntoError> {
        match target {
            Mixed::Real(d) => self.onto(item, d).map(MixedTypeDom::Real),
            Mixed::Nat(d) => self.onto(item, d).map(MixedTypeDom::Nat),
            Mixed::Int(d) => self.onto(item, d).map(MixedTypeDom::Int),
            Mixed::Bool(d) => self.onto(item, d).map(MixedTypeDom::Bool),
            Mixed::Unit(d) => self.onto(item, d).map(MixedTypeDom::Unit),
            Mixed::GridReal(d) => self.onto(item, d).map(MixedTypeDom::GridReal),
            Mixed::GridNat(d) => self.onto(item, d).map(MixedTypeDom::GridNat),
            Mixed::GridInt(d) => self.onto(item, d).map(MixedTypeDom::GridInt),
            Mixed::Cat(d) => self.onto(item, d).map(MixedTypeDom::Cat),
        }
    }
}
impl<In> OntoDom<Mixed> for Bounded<In>
where
    In: BoundedBounds,
    f64: AsPrimitive<In>,
{
}

/// [`Bounded`] alias for a continuous `f64` [`Domain`] bounded by a lower and upper bounds.
///
/// # Attributes
///
/// * `lower` - Lower bound of the [`Real`] [`Domain`].
/// * `upper` - Upper bound of the [`Real`] [`Domain`].
/// * `mid` - Middle point of the [`Real`] [`Domain`].
/// * `width` - Width of the [`Real`] [`Domain`].
///
/// # Examples
///
/// ```
/// use tantale::core::{Real,Domain,Uniform};
/// let dom = Real::new(0.0, 10.0,Uniform);
///
/// let mut rng = rand::rng();
/// let sample = dom.sample(&mut rng);
/// assert!(dom.is_in(&sample));
/// assert_eq!(*dom.bounds.start(), 0.0);
/// assert_eq!(*dom.bounds.end(), 10.0);
/// assert_eq!(dom.mid, 5.0);
/// assert_eq!(dom.width, 10.0);
/// ```
pub type Real = Bounded<f64>;

/// [`Bounded`] alias for a natural `u64` [`Domain`] bounded by a lower and upper bounds.
///
/// # Attributes
///
/// * `lower` - Lower bound of the [`Nat`] [`Domain`].
/// * `upper` - Upper bound of the [`Nat`] [`Domain`].
/// * `mid` - Middle point of the [`Nat`] [`Domain`].
/// * `width` - Width of the [`Nat`] [`Domain`].
///
/// # Examples
///
/// ```
/// use tantale::core::{Nat,Domain,Uniform};
/// let dom = Nat::new(0, 10, Uniform);
///
/// let mut rng = rand::rng();
/// let sample = dom.sample(&mut rng);
/// assert!(dom.is_in(&sample));
/// assert_eq!(*dom.bounds.start(), 0);
/// assert_eq!(*dom.bounds.end(), 10);
/// assert_eq!(dom.mid, 5);
/// assert_eq!(dom.width, 10);
/// ```
pub type Nat = Bounded<u64>;

/// [`Bounded`] alias for an integer `i64` [`Domain`] bounded by a lower and upper bounds.
///
/// # Attributes
///
/// * `lower` - Lower bound of the [`Int`] [`Domain`].
/// * `upper` - Upper bound of the [`Int`] [`Domain`].
/// * `mid` - Middle point of the [`Int`] [`Domain`].
/// * `width` - Width of the [`Int`] [`Domain`].
///
/// # Examples
///
/// ```
/// use tantale::core::{Int,Domain,Uniform};
///
/// let dom = Int::new(0, 10, Uniform);
///
/// let mut rng = rand::rng();
/// let sample = dom.sample(&mut rng);
/// assert!(dom.is_in(&sample));
/// assert_eq!(*dom.bounds.start(), 0);
/// assert_eq!(*dom.bounds.end(), 10);
/// assert_eq!(dom.mid, 5);
/// assert_eq!(dom.width, 10);
/// ```
pub type Int = Bounded<i64>;

impl<T> CSVWritable<(), <Bounded<T> as Domain>::TypeDom> for Bounded<T>
where
    T: BoundedBounds,
{
    fn header(_elem: &()) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, comp: &<Bounded<T> as Domain>::TypeDom) -> Vec<String> {
        Vec::from([comp.to_string()])
    }
}
