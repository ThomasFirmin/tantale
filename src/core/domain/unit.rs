use crate::core::domain::Domain;
use crate::core::domain::bounded::{DomainBounded, Bounded,BoundedBounds};
use crate::core::domain::bool::Bool;
use crate::core::domain::cat::Cat;
use crate::core::domain::base::{BaseDom,BaseTypeDom};
use crate::core::domain::sampler::uniform;
use crate::core::domain::onto::Onto;
use crate::core::domain::errors_domain::{DomainError,DomainOoBError};

use std::ops::RangeInclusive;
use std::fmt;
use num::cast::AsPrimitive;
use num::Float;
use rand::rngs::ThreadRng;

pub trait UnitBounds : BoundedBounds + Float {}
impl<T:BoundedBounds + Float> UnitBounds for T {}


/// A [`Unit`] domain within `[0,1]`. The floating point type is inferred.
/// /// A generic [`Unit`] [`Domain`] with a numerical `lower=0.0` and `upper=1.0` bounds.
///
pub struct Unit<T:UnitBounds>
{
    bounds: RangeInclusive<T>,
}

impl<T:UnitBounds> Unit<T>
{
    /// Fabric for a [`Unit`] [`Domain`].
    pub fn new() -> Unit<T> {
        Unit {
            bounds: RangeInclusive::new(T::from(0.0).unwrap(), T::from(1.0).unwrap()),
        }
    }
}

impl<T:UnitBounds> Domain for Unit<T>
{
    type TypeDom = T;

    /// Default sampler for [`Unit`].
    /// See [`uniform`].
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        |s, rng| uniform(&s.bounds, rng)
    }

    fn is_in(&self, item: &Self::TypeDom) -> bool {
        self.bounds.contains(item)
    }
}

impl<T:UnitBounds> DomainBounded for Unit<T>
{
    fn lower(&self) -> Self::TypeDom {
        T::from(0.0).unwrap()
    }
    fn upper(&self) -> Self::TypeDom {
        T::from(1.0).unwrap()
    }
    fn mid(&self) -> Self::TypeDom {
        Self::TypeDom::from(0.5).unwrap()
    }
    fn width(&self) -> Self::TypeDom {
        T::from(1.0).unwrap()
    }
}

impl<T:UnitBounds> std::clone::Clone for Unit<T>
{
    fn clone(&self) -> Self {
        Unit::new()
    }
}

impl<T:UnitBounds> fmt::Display for Unit<T>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}

impl<T:UnitBounds> fmt::Debug for Unit<T>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}


impl<In, Out> Onto<Bounded<Out>> for Unit<In>
where
    In : UnitBounds,
    Out:BoundedBounds,
    f64: AsPrimitive<Out>,
{
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
            let a: f64 = (*item).as_();
            let c: f64 = target.width().as_();
            let mapped: Out = (a * c).as_() + target.lower();

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

impl<In> Onto<Bool> for Unit<In>
where
    In: UnitBounds,
    f64: AsPrimitive<In>,
{
    /// [`Onto`] function between a [`Unit`] [`Domain`] and a [`Bool`][`Domain`].
    ///
    /// Considering $x$ the `item`, returns `true` if $x>0.5$
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Bool`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &In, _target: &Bool) -> Result<<Bool as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            Ok(*item > In::from(0.5).unwrap())
        } else {
            Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, self))))
        }
    }
}

impl<'a, In, const N: usize> Onto<Cat<'a, N>> for Unit<In>
where
    In: UnitBounds,
    f64: AsPrimitive<In>,
{
    /// [`Onto`] function between a [`Unit`] [`Domain`] and a [`Cat`][`Domain`].
    ///
    /// Considering $\ell_{en}$ the length of `values` of the [`Cat`] [`Domain`].
    /// The variable $x$ is the item to be mapped.
    /// The mapping is given by mapping item to an index of `values` in [`Cat`]:
    ///
    /// $$ \texttt{Floor}\Biggl(x \times (\ell_{en}-1)\Biggl) $$
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
            let a: f64 = item.as_();
            let c: f64 = (target.values().len() - 1).as_();
            let mapped = target.values()[(a* c) as usize];

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

impl<'a, In, const N :usize,T,U> Onto<BaseDom<'a,N,T,U>> for Unit<In>
where
    In:UnitBounds,
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
            BaseDom::Unit(_d) => unreachable!("Converting a value from Unit onto Unit is not implemented, and it should not occur."),
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