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
//! use tantale::core::{Bounded, Domain, DomainBounded};
//! let dom : Bounded<u8> = Bounded::new(0, 255);
//!
//! let mut rng = rand::rng();
//! let sample = dom.sample(&mut rng);
//! assert!(dom.is_in(&sample));
//! assert_eq!(dom.lower(), 0);
//! assert_eq!(dom.upper(), 255);
//! assert_eq!(dom.mid(), 127);
//! assert_eq!(dom.width(), 255);
//! ```

use crate::domain::{
    base::{BaseDom, BaseTypeDom},
    bool::Bool,
    cat::Cat,
    derrors::{DomainError, DomainOoBError},
    onto::Onto,
    sampler::uniform,
    unit::Unit,
    Domain,
};

use num::{cast::AsPrimitive, Num, NumCast};
use rand::{distr::uniform::SampleUniform, prelude::ThreadRng};
use std::{
    fmt::{self, Debug, Display},
    ops::RangeInclusive,
};

// _-_-_-_-_-_-__-_-_-_-_-_-_-_
// Bounded domain


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
{
}

/// Describes a peculiar trait for some of the domains that are numerically bounded by a
/// `lower` and `upper` bound.
pub trait DomainBounded: Domain {
    /// Getter method for the lower bound.
    fn lower(&self) -> Self::TypeDom;
    /// Getter method for the upper bound.
    fn upper(&self) -> Self::TypeDom;
    /// Getter method for the middle point of the domain.
    fn mid(&self) -> Self::TypeDom;
    /// Getter method for the width of the domain.
    fn width(&self) -> Self::TypeDom;
}

/// A generic [`Bounded`] [`Domain`] with a numerical `lower` and `upper` bounds, modeled
/// by a [`RangeInclusive`].
///
/// # Attributes
/// * `bounds`: [`RangeInclusive`]`<T>` - A [`RangeInclusive`] object of type `<T>`.
/// * `mid`: `T` - Middle point of the [`Bounded`] [`Domain`]. $\frac{\texttt{lower}+\texttt{upper}}{2}$
/// * `width`: `T` - Width of the [`Bounded`] [`Domain`]. $\texttt{upper}-\texttt{lower}$
///
pub struct Bounded<T: BoundedBounds> {
    pub bounds: RangeInclusive<T>,
    pub mid: T,
    pub width: T,
}

impl<T: BoundedBounds> Bounded<T> {
    /// Fabric for a [`Bounded`].
    ///
    /// * The `mid` attribute is automatically computed with $\frac{\texttt{lower}+\texttt{upper}}{2}$.
    /// * The `width` attribute is automatically computed with $\texttt{upper}-\texttt{lower}$.
    ///
    /// # Parameters
    /// * `lower`: `T` - Lower bound of the [`Bounded`] [`Domain`].
    /// * `upper`: `T` - Upper bound of the [`Bounded`] [`Domain`].
    ///
    pub fn new(lower: T, upper: T) -> Bounded<T> {
        if lower < upper {
            let mid = (upper.clone() + lower.clone()) / T::from(2).unwrap();
            let width = upper.clone() - lower.clone();
            Bounded {
                bounds: std::ops::RangeInclusive::new(lower, upper),
                mid,
                width,
            }
        } else {
            panic!("Boundaries error, {} is not < {}.", lower, upper);
        }
    }
}

impl<T: BoundedBounds> PartialEq for Bounded<T> {
    fn eq(&self, other: &Self) -> bool {
        (self.lower() == other.lower()) && (self.upper() == other.upper())
    }
}

impl<T: BoundedBounds> Domain for Bounded<T> {
    type TypeDom = T;

    /// Default sampler for [`Bounded`].
    /// See [`uniform`].
    fn sample(&self, rng: &mut ThreadRng) -> Self::TypeDom {
        uniform(&self.bounds, rng)
    }

    fn is_in(&self, item: &T) -> bool {
        self.bounds.contains(item)
    }
}

impl<T: BoundedBounds> DomainBounded for Bounded<T> {
    fn lower(&self) -> Self::TypeDom {
        self.bounds.start().clone()
    }
    fn upper(&self) -> Self::TypeDom {
        self.bounds.end().clone()
    }
    fn mid(&self) -> Self::TypeDom {
        self.mid.clone()
    }
    fn width(&self) -> Self::TypeDom {
        self.width.clone()
    }
}

impl<T: BoundedBounds> std::clone::Clone for Bounded<T> {
    fn clone(&self) -> Self {
        Bounded::new(self.bounds.start().clone(), self.bounds.end().clone())
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

impl<In, Out> Onto<Bounded<Out>> for Bounded<In>
where
    In: BoundedBounds,
    Out: BoundedBounds,
    f64: AsPrimitive<Out>,
{
    /// [`Onto`] function between [`Bounded`] [`Domain`].
    ///
    /// Considering $l_{in}$, $u_{in}$, $l_{out}$ and $u_{out}$ the lower and upper bounds of
    /// the input [`Bounded`] [`Domain`] and the output [`Bounded`] [`Domain`], and $x$ the item to be mapped,
    /// the mapping is given by
    ///
    /// $$ \frac{x-l_{in}}{u_{in}-l_{in}} \times (u_{out}-l_{out}) + l_{out}$$
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Bounded`]`<Out>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &In, target: &Bounded<Out>) -> Result<Out, DomainError> {
        if self.is_in(item) {
            let a: f64 = (*item - self.lower()).as_();
            let b: f64 = self.width().as_();
            let c: f64 = target.width().as_();
            let mapped: Out = (a / b * c).as_() + target.lower();

            if target.is_in(&mapped) {
                Ok(mapped)
            } else {
                Err(DomainError::OoB(DomainOoBError(format!(
                    "{} -> {} mapped input not in {}",
                    item, mapped, target
                ))))
            }
        } else {
            Err(DomainError::OoB(DomainOoBError(format!(
                "{} input not in {}",
                item, self
            ))))
        }
    }
}

impl<In> Onto<Bool> for Bounded<In>
where
    In: BoundedBounds,
    f64: AsPrimitive<In>,
{
    /// [`Onto`] function between a [`Bounded`] [`Domain`] and a [`Bool`][`Domain`].
    ///
    /// Considering $l_{in}$ and $u_{in}$ the lower and upper bounds of
    /// the input [`Bounded`] [`Domain`], and $x$ the `item`, returns `true` if $x>\frac{u_{in}-l_{in}}{2}$
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Bool`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainOoBError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &In, _target: &Bool) -> Result<<Bool as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            Ok(*item > self.mid())
        } else {
            Err(DomainError::OoB(DomainOoBError(format!(
                "{} input not in {}",
                item, self
            ))))
        }
    }
}

impl<'a, In> Onto<Cat> for Bounded<In>
where
    In: BoundedBounds,
    f64: AsPrimitive<In>,
{
    /// [`Onto`] function between a [`Bounded`] [`Domain`] and a [`Cat`][`Domain`].
    ///
    /// Considering $l_{in}$, $u_{in}$ the lower and upper bounds of
    /// the input [`Bounded`] [`Domain`] and $\ell_{en}$ the length of `values` of the [`Cat`] [`Domain`].
    /// The variable $x$ is the item to be mapped.
    /// The mapping is given by mapping item to an index of `values` in [`Cat`]:
    ///
    /// $$ i = \\left\\lfloor \frac{x-l_{in}}{u_{in}-l_{in}} \times \ell_{en} \\right\\rfloor $$
    /// $$ \\texttt{index} = \\begin{cases}
    ///    i & \\text{if } i < \ell_{en} \\\\
    ///    i -1 & \\text{if } i = \ell_{en}
    /// \\end{cases} $$
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Cat`]<'a> - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &In, target: &Cat) -> Result<<Cat as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let a: f64 = (*item - self.lower()).as_();
            let b: f64 = self.width().as_();
            let c: f64 = target.values().len().as_();
            let idx = (a / b * c) as usize;
            let idx = if idx == target.values().len(){idx-1}else{idx};
            let mapped = target.values()[idx];

            if target.is_in(&mapped) {
                Ok(mapped)
            } else {
                Err(DomainError::OoB(DomainOoBError(format!(
                    "{} -> {} mapped input not in {}",
                    item, mapped, target
                ))))
            }
        } else {
            Err(DomainError::OoB(DomainOoBError(format!(
                "{} input not in {}",
                item, self
            ))))
        }
    }
}

impl<In> Onto<Unit> for Bounded<In>
where
    In: BoundedBounds,
{
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
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Unit`]`<Out>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &In, target: &Unit) -> Result<f64, DomainError> {
        if self.is_in(item) {
            let a: f64 = (*item - self.lower()).as_();
            let b: f64 = self.width().as_();
            let mapped: f64 = a / b;

            if target.is_in(&mapped) {
                Ok(mapped)
            } else {
                Err(DomainError::OoB(DomainOoBError(format!(
                    "{} -> {} mapped input not in {}",
                    item, mapped, target
                ))))
            }
        } else {
            Err(DomainError::OoB(DomainOoBError(format!(
                "{} input not in {}",
                item, self
            ))))
        }
    }
}

impl<'a, In> Onto<BaseDom> for Bounded<In>
where
    In: BoundedBounds,
    f64: AsPrimitive<In>,
{
    /// [`Onto`] function between a [`Bounded`] [`Domain`] and a [`BaseDom`][`Domain`].
    ///
    //// Match a targetted[`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and (`onto`)[`Onto::onto`] of [`Self`].
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`BaseDom`]`<'a>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &In,
        target: &BaseDom,
    ) -> Result<<BaseDom as Domain>::TypeDom, DomainError> {
        match target {
            BaseDom::Real(d) => match self.onto(item, d) {
                Ok(i) => Ok(BaseTypeDom::Real(i)),
                Err(e) => Err(e),
            },
            BaseDom::Nat(d) => match self.onto(item, d) {
                Ok(i) => Ok(BaseTypeDom::Nat(i)),
                Err(e) => Err(e),
            },
            BaseDom::Int(d) => match self.onto(item, d) {
                Ok(i) => Ok(BaseTypeDom::Int(i)),
                Err(e) => Err(e),
            },
            BaseDom::Unit(d) => match self.onto(item, d) {
                Ok(i) => Ok(BaseTypeDom::Unit(i)),
                Err(e) => Err(e),
            },
            BaseDom::Bool(d) => match self.onto(item, d) {
                Ok(i) => Ok(BaseTypeDom::Bool(i)),
                Err(e) => Err(e),
            },
            BaseDom::Cat(d) => match self.onto(item, d) {
                Ok(i) => Ok(BaseTypeDom::Cat(i)),
                Err(e) => Err(e),
            },
        }
    }
}

impl From<BaseDom> for Real {
    fn from(value: BaseDom) -> Self {
        match value {
            BaseDom::Real(d) => d,
            _ => unreachable!("Can only use From<BaseDom> with Bounded."),
        }
    }
}
impl From<BaseDom> for Nat {
    fn from(value: BaseDom) -> Self {
        match value {
            BaseDom::Nat(d) => d,
            _ => unreachable!("Can only use From<BaseDom> with Bounded."),
        }
    }
}
impl From<BaseDom> for Int {
    fn from(value: BaseDom) -> Self {
        match value {
            BaseDom::Int(d) => d,
            _ => unreachable!("Can only use From<BaseDom> with Bounded."),
        }
    }
}
/// [`Bounded`] alias for a continuous `f64` [`Domain`] bounded by a lower and upper bounds.
///
/// # Attributes
///
/// * `lower`: `f64` - Lower bound of the [`Real`] [`Domain`].
/// * `upper`: `f64` - Upper bound of the [`Real`] [`Domain`].
/// * `mid`: `f64` - Middle point of the [`Real`] [`Domain`].
/// * `width`: `f64` - Width of the [`Real`] [`Domain`].
///
/// # Examples
///
/// ```
/// use tantale::core::{Real,Domain,DomainBounded};
/// let dom = Real::new(0.0, 10.0);
///
/// let mut rng = rand::rng();
/// let sample = dom.sample(&mut rng);
/// assert!(dom.is_in(&sample));
/// assert_eq!(dom.lower(), 0.0);
/// assert_eq!(dom.upper(), 10.0);
/// assert_eq!(dom.mid(), 5.0);
/// assert_eq!(dom.width(), 10.0);
/// ```
pub type Real = Bounded<f64>;

/// [`Bounded`] alias for a natural `u64` [`Domain`] bounded by a lower and upper bounds.
///
/// # Attributes
///
/// * `lower`: `u64` - Lower bound of the [`Nat`] [`Domain`].
/// * `upper`: `u64` - Upper bound of the [`Nat`] [`Domain`].
/// * `mid`: `u64` - Middle point of the [`Nat`] [`Domain`].
/// * `width`: `u64` - Width of the [`Nat`] [`Domain`].
///
/// # Examples
///
/// ```
/// use tantale::core::{Nat,Domain,DomainBounded};
/// let dom = Nat::new(0, 10);
///
/// let mut rng = rand::rng();
/// let sample = dom.sample(&mut rng);
/// assert!(dom.is_in(&sample));
/// assert_eq!(dom.lower(), 0);
/// assert_eq!(dom.upper(), 10);
/// assert_eq!(dom.mid(), 5);
/// assert_eq!(dom.width(), 10);
/// ```
pub type Nat = Bounded<u64>;

/// [`Bounded`] alias for an integer `i64` [`Domain`] bounded by a lower and upper bounds.
///
/// # Attributes
///
/// * `lower`: `i64` - Lower bound of the [`Int`] [`Domain`].
/// * `upper`: `i64` - Upper bound of the [`Int`] [`Domain`].
/// * `mid`: `i64` - Middle point of the [`Int`] [`Domain`].
/// * `width`: `i64` - Width of the [`Int`] [`Domain`].
///
/// # Examples
///
/// ```
/// use tantale::core::{Int,Domain,DomainBounded};
///
/// let dom = Int::new(0, 10);
///
/// let mut rng = rand::rng();
/// let sample = dom.sample(&mut rng);
/// assert!(dom.is_in(&sample));
/// assert_eq!(dom.lower(), 0);
/// assert_eq!(dom.upper(), 10);
/// assert_eq!(dom.mid(), 5);
/// assert_eq!(dom.width(), 10);
/// ```
pub type Int = Bounded<i64>;
