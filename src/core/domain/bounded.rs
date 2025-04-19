use crate::core::domain::Domain;
use crate::core::domain::bool::Bool;
use crate::core::domain::cat::Cat;
use crate::core::domain::unit::{Unit,UnitBounds};
use crate::core::domain::base::{BaseDom, BaseTypeDom};
use crate::core::domain::sampler::uniform;
use crate::core::domain::onto::Onto;
use crate::core::domain::errors_domain::{DomainError,DomainOoBError};

use std::ops::RangeInclusive;
use std::fmt::{self, Debug, Display};
use num::cast::AsPrimitive;
use num::{Num, NumCast};
use rand::rngs::ThreadRng;
use rand::distr::uniform::SampleUniform;


// _-_-_-_-_-_-__-_-_-_-_-_-_-_
// Bounded domain

pub trait BoundedBounds : Num + NumCast + PartialEq + PartialOrd + Copy + Clone + SampleUniform + AsPrimitive<f64> + Display + Debug {}
impl<T> BoundedBounds for T 
where
    T: Num + NumCast + PartialEq + PartialOrd + Copy + Clone + SampleUniform + AsPrimitive<f64> + Display + Debug,
{}

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

/// A generic [`Bounded`] [`Domain`] with a numerical `lower` and `upper` bounds.
///
/// # Attributes
/// * `bounds`: [`RangeInclusive`]`<T>` - A [`RangeInclusive`] object of type `<T>`.
/// * `mid`: `T` - Middle point of the [`Bounded`] [`Domain`]. $\frac{\texttt{lower}+\texttt{upper}}{2}$
/// * `width`: `T` - Width of the [`Bounded`] [`Domain`]. $\texttt{upper}-\texttt{lower}$
///
pub struct Bounded<T:BoundedBounds>
{
    bounds: RangeInclusive<T>,
    mid: T,
    width: T,
}

impl<T: BoundedBounds> Bounded<T>
{
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

impl<T:BoundedBounds> Domain for Bounded<T>
{
    type TypeDom = T;

    /// Default sampler for [`Bounded`].
    /// See [`uniform`].
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        |d, rng| uniform(&d.bounds, rng)
    }

    fn is_in(&self, item: &T) -> bool {
        self.bounds.contains(item)
    }
}

impl<T:BoundedBounds> DomainBounded for Bounded<T>
{
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

impl<T:BoundedBounds> std::clone::Clone for Bounded<T>
{
    fn clone(&self) -> Self {
        Bounded::new(self.bounds.start().clone(), self.bounds.end().clone())
    }
}

impl<T:BoundedBounds> fmt::Display for Bounded<T>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}

impl<T:BoundedBounds> fmt::Debug for Bounded<T>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}


impl<In, Out> Onto<Bounded<Out>> for Bounded<In>
where
    In : BoundedBounds,
    Out : BoundedBounds,
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
                Err(DomainError::OoB(DomainOoBError(format!("{} -> {} mapped input not in {}", item, mapped, target))))
            }
        } else {
            Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, self))))
        }
    }
}

impl<In> Onto<Bool> for Bounded<In>
where
    In:BoundedBounds,
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
            Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, self))))
        }
    }
}

impl<'a, In, const N: usize> Onto<Cat<'a, N>> for Bounded<In>
where
    In : BoundedBounds,
    f64: AsPrimitive<In>,
{
    /// [`Onto`] function between a [`Bounded`] [`Domain`] and a [`Cat`][`Domain`].
    ///
    /// Considering $l_{in}$, $u_{in}$ the lower and upper bounds of
    /// the input [`Bounded`] [`Domain`] and $\ell_{en}$ the length of `values` of the [`Cat`] [`Domain`].
    /// The variable $x$ is the item to be mapped.
    /// The mapping is given by mapping item to an index of `values` in [`Cat`]:
    ///
    /// $$ \texttt{Floor}\Biggl(\frac{x-l_{in}}{u_{in}-l_{in}} \times (\ell_{en}-1)\Biggl) $$
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Cat`]<'a, N> - A borrowed targetted [`Domain`].
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
        target: &Cat<'a, N>,
    ) -> Result<<Cat<'a, N> as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let a: f64 = (*item - self.lower()).as_();
            let b: f64 = self.width().as_();
            let c: f64 = (target.values().len() - 1).as_();
            let mapped = target.values()[(a / b * c) as usize];

            if target.is_in(&mapped) {
                Ok(mapped)
            } else {
                Err(DomainError::OoB(DomainOoBError(format!("{} -> {} mapped input not in {}", item, mapped, target))))
            }
        } else {
            Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, self))))
        }
    }
}

impl<In, Out> Onto<Unit<Out>> for Bounded<In>
where
    In:BoundedBounds,
    Out:UnitBounds,
    f64: AsPrimitive<Out>,
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
    fn onto(&self, item: &In, target: &Unit<Out>) -> Result<Out, DomainError> {
        if self.is_in(item) {
            let a: f64 = (*item - self.lower()).as_();
            let b: f64 = self.width().as_();
            let mapped: Out = (a / b).as_();

            if target.is_in(&mapped) {
                Ok(mapped)
            } else {
                Err(DomainError::OoB(DomainOoBError(format!("{} -> {} mapped input not in {}", item, mapped, target))))
            }
        } else {
            Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, self))))
        }
    }
}

impl<'a, In, const N :usize,T,U> Onto<BaseDom<'a,N,T,U>> for Bounded<In>
where
    In:BoundedBounds,
    T:BoundedBounds,
    U:UnitBounds,
    f64: AsPrimitive<In>,
    f64: AsPrimitive<T>,
    f64: AsPrimitive<U>,
{
    /// [`Onto`] function between a [`Bounded`] [`Domain`] and a [`BaseDom`][`Domain`].
    ///
    //// Match a targetted[`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and (`onto`)[`Onto::onto`] of [`Self`].
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`BaseDom`]`<'a,N,T,U>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &In, target: &BaseDom<'a,N,T,U>) -> Result<<BaseDom<'a,N,T,U> as Domain>::TypeDom, DomainError> {
        match target{
            BaseDom::Bounded(d) => {
                match self.onto(item, d) {
                    Ok(i) => Ok(BaseTypeDom::<'a,N,T,U>::Bounded(i)),
                    Err(e) => Err(e),
                }
            },
            BaseDom::Unit(d) => {
                match self.onto(item, d) {
                    Ok(i) => Ok(BaseTypeDom::<'a,N,T,U>::Unit(i)),
                    Err(e) => Err(e),
                }
            },
            BaseDom::Bool(d) => {
                match self.onto(item, d) {
                    Ok(i) => Ok(BaseTypeDom::<'a,N,T,U>::Bool(i)),
                    Err(e) => Err(e),
                }
            },
            BaseDom::Cat(d) => {
                match self.onto(item, d) {
                    Ok(i) => Ok(BaseTypeDom::<'a,N,T,U>::Cat(i)),
                    Err(e) => Err(e),
                }
            },
        }
    }
}

/// [`Bounded`] alias for a continuous `f64` [`Domain`] bounded by a lower and upper bounds.
///
/// # Attributes
///
/// For safety, the attributes are private and can only be accessed via getter methods
/// `.lower()`, `.upper()`, `.mid()` and `.width()`.
/// It prevents any modification of the [`Domain`] during the optimization process.
/// Some algorithms might modify the [`Domain`], for example *divide-and-conquer* approaches.
/// In that case, the algorithm should create a new [`Domain`].
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
/// let sampler = dom.default_sampler();
/// assert!(dom.is_in(&sampler(&dom, &mut rng)));
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
/// For safety, the attributes are private and can only be accessed via getter methods
/// `.lower()`, `.upper()`, `.mid()` and `.width()`.
/// It prevents any modification of the [`Domain`] during the optimization process.
/// Some algorithms might modify the [`Domain`], for example *divide-and-conquer* approaches.
/// In that case, the algorithm should create a new [`Domain`].
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
/// let sampler = dom.default_sampler();
/// assert!(dom.is_in(&sampler(&dom, &mut rng)));
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
/// For safety, the attributes are private and can only be accessed via getter methods
/// `.lower()`, `.upper()`, `.mid()` and `.width()`.
/// It prevents any modification of the [`Domain`] during the optimization process.
/// Some algorithms might modify the [`Domain`], for example *divide-and-conquer* approaches.
/// In that case, the algorithm should create a new [`Domain`].
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
/// let sampler = dom.default_sampler();
/// assert!(dom.is_in(&sampler(&dom, &mut rng)));
/// assert_eq!(dom.lower(), 0);
/// assert_eq!(dom.upper(), 10);
/// assert_eq!(dom.mid(), 5);
/// assert_eq!(dom.width(), 10);
/// ```
pub type Int = Bounded<i64>;