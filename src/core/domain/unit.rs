use crate::core::domain::Domain;
use crate::core::domain::bounded::{DomainBounded, Bounded,BoundedBounds};
use crate::core::domain::bool::Bool;
use crate::core::domain::cat::Cat;
use crate::core::domain::base::{BaseDom,BaseTypeDom};
use crate::core::domain::sampler::uniform;
use crate::core::domain::onto::Onto;
use crate::core::domain::errors_domain::{DomainError,DomainOoBError};

use std::fmt;
use std::ops::RangeInclusive;
use num::cast::AsPrimitive;
use rand::rngs::ThreadRng;

/// A [`f64`] [`Unit`] domain within `[0,1]`.
/// /// A generic [`Unit`] [`Domain`] with a numerical `lower=0.0` and `upper=1.0` bounds.
///
pub struct Unit
{
    bounds: RangeInclusive<f64>,
}

impl Unit
{
    /// Fabric for a [`Unit`] [`Domain`].
    pub fn new() -> Unit {
        Unit {
            bounds: RangeInclusive::new(0.0, 1.0),
        }
    }
}

impl PartialEq for Unit{
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Domain for Unit
{
    type TypeDom = f64;

    /// Default sampler for [`Unit`].
    /// See [`uniform`].
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        |s, rng| uniform(&s.bounds, rng)
    }

    fn is_in(&self, item: &Self::TypeDom) -> bool {
        self.bounds.contains(item)
    }
}

impl DomainBounded for Unit
{
    fn lower(&self) -> Self::TypeDom {
        0.0
    }
    fn upper(&self) -> Self::TypeDom {
        1.0
    }
    fn mid(&self) -> Self::TypeDom {
        0.5
    }
    fn width(&self) -> Self::TypeDom {
        1.0
    }
}

impl std::clone::Clone for Unit
{
    fn clone(&self) -> Self {
        Unit::new()
    }
}

impl fmt::Display for Unit
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}

impl fmt::Debug for Unit
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}


impl<Out> Onto<Bounded<Out>> for Unit
where
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
    fn onto(&self, item: &f64, target: &Bounded<Out>) -> Result<Out, DomainError> {
        if self.is_in(item) {
            let a: f64 = *item;
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

impl Onto<Bool> for Unit
where
    f64: AsPrimitive<f64>,
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
    fn onto(&self, item: &f64, _target: &Bool) -> Result<<Bool as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            Ok(*item > 0.5)
        } else {
            Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, self))))
        }
    }
}

impl<'a> Onto<Cat<'a>> for Unit
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
        item: &f64,
        target: &Cat<'a>,
    ) -> Result<<Cat<'a> as Domain>::TypeDom, DomainError> {
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

impl<'a> Onto<BaseDom<'a>> for Unit
{
    /// [`Onto`] function between a [`Bounded`] [`Domain`] and a [`BaseDom`][`Domain`].
    ///
    //// Match a targetted[`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and (`onto`)[`Onto::onto`] of [`Self`].
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`BaseDom`]`<'a,N,T>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &f64, target: &BaseDom<'a>) -> Result<<BaseDom<'a> as Domain>::TypeDom, DomainError> {
        match target{
            BaseDom::Real(d) => {
                match self.onto(item, d) {
                    Ok(i) => Ok(BaseTypeDom::Real(i)),
                    Err(e) => Err(e),
                }
            },
            BaseDom::Nat(d) => {
                match self.onto(item, d) {
                    Ok(i) => Ok(BaseTypeDom::Nat(i)),
                    Err(e) => Err(e),
                }
            },
            BaseDom::Int(d) => {
                match self.onto(item, d) {
                    Ok(i) => Ok(BaseTypeDom::Int(i)),
                    Err(e) => Err(e),
                }
            },
            BaseDom::Unit(_d) => unreachable!("Converting a value from Unit onto Unit is not implemented, and it should not occur."),
            BaseDom::Bool(d) => {
                match self.onto(item, d) {
                    Ok(i) => Ok(BaseTypeDom::Bool(i)),
                    Err(e) => Err(e),
                }
            },
            BaseDom::Cat(d) => {
                match self.onto(item, d) {
                    Ok(i) => Ok(BaseTypeDom::Cat(i)),
                    Err(e) => Err(e),
                }
            },
        }
    }
}

impl Onto<Unit> for Unit
{
    /// [`Onto`] function between a [`Unit`] and another [`Unit`] [`Domain`].
    ///
    /// Returns its cloned input [`item`].
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Unit`] - A borrowed targetted [`Domain`].
    ///
    fn onto(
        &self,
        item: &<Unit as Domain>::TypeDom,
        _target: &Unit,
    ) -> Result<f64, DomainError> {
        Ok(item.clone())
    }
}

impl <'a> From<BaseDom<'a>> for Unit
{
    fn from(value: BaseDom<'a>) -> Self {
        match value{
            BaseDom::Unit(d)=>d,
            _ => unreachable!("Can only From<BaseDom> with Unit.")
        }
    }
}