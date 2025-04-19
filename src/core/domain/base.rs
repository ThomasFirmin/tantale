use crate::core::domain::Domain;
use crate::core::domain::bounded::{Bounded,BoundedBounds};
use crate::core::domain::unit::{Unit,UnitBounds};
use crate::core::domain::bool::Bool;
use crate::core::domain::cat::Cat;
use crate::core::domain::errors_domain::{DomainError,DomainOoBError};
use crate::core::domain::onto::Onto;

use num::cast::AsPrimitive;
use rand::prelude::ThreadRng;
use std::fmt::{Debug, Display};

// -_-_-_-_-_-_-_-
// Grouped domains

/// A [`enum`] [`BaseDom`] [`Domain`], made of the 4 basic domains [`Bounded`], [`Bool`], [`Cat`] and [`Unit`].
/// Used for mixed [`Element`] and [`Solution`]. The (`TypeDom`)[`Domain::TypeDom`] is an [`enum`] [`BaseTypeDom`].
///
pub enum BaseDom<'a,const N : usize,T,U>
where
    T: BoundedBounds,
    U : UnitBounds,
{
    Bounded(Bounded<T>),
    Bool(Bool),
    Cat(Cat<'a,N>),
    Unit(Unit<U>),
}

/// Basic (`TypeDom`)[`Domain::TypeDom`] of [`BaseDom`].
/// 
#[derive(Copy,Clone,PartialEq)]
pub enum BaseTypeDom<'a, const N : usize,T, U>
where
    T: BoundedBounds,
    U : UnitBounds
{
    Bounded(<Bounded<T> as Domain>::TypeDom),
    Bool(<Bool as Domain>::TypeDom),
    Cat(<Cat<'a,N> as Domain>::TypeDom),
    Unit(<Unit<U> as Domain>::TypeDom),
}

impl<'a, const N:usize, T, U> Display for BaseTypeDom<'a, N, T, U>
where
    T : BoundedBounds,
    U : UnitBounds,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {

        match self {
            Self::Bounded(d) => std::fmt::Display::fmt(&d, f),
            Self::Unit(d) => std::fmt::Display::fmt(&d, f),
            Self::Bool(d) => std::fmt::Display::fmt(&d, f),
            Self::Cat(d) => std::fmt::Display::fmt(&d, f),
        }
    }
}
impl<'a, const N:usize,T,U> Debug for BaseTypeDom<'a, N,T,U>
where
    T: BoundedBounds,
    U: UnitBounds,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {

        match self {
            Self::Bounded(d) => std::fmt::Debug::fmt(&d, f),
            Self::Unit(d) => std::fmt::Debug::fmt(&d, f),
            Self::Bool(d) => std::fmt::Debug::fmt(&d, f),
            Self::Cat(d) => std::fmt::Debug::fmt(&d, f),
        }
    }
}

impl <'a, const N : usize,T,U> Domain for BaseDom<'a, N,T,U>
where
    T : BoundedBounds,
    U:UnitBounds,
{
    type TypeDom = BaseTypeDom<'a,N,T,U>;
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        |d,rng| d.default_sampler()(d,rng)
    }

    fn is_in(&self, item: &Self::TypeDom) -> bool {
        match self {
            Self::Bounded(d) => match item{
                Self::TypeDom::Bounded(i) => d.is_in(i),
                _ => false,
            },
            Self::Unit(d) => match item{
                Self::TypeDom::Unit(i) => d.is_in(i),
                _ => false,
            },
            Self::Bool(d) => match item{
                Self::TypeDom::Bool(i) => d.is_in(i),
                _ => false,
            },
            Self::Cat(d) => match item{
                Self::TypeDom::Cat(i) => d.is_in(i),
                _ => false,
            },
        }
    }
}

impl<'a, const N: usize, T, U, Out> Onto<Bounded<Out>> for BaseDom<'a, N, T, U>
where
    T : BoundedBounds,
    U : UnitBounds,
    Out: BoundedBounds,
    f64: AsPrimitive<Out>
{
    /// [`Onto`] function between a [`BaseDom`] and a [`Bounded`] [`Domain`].
    ///
    /// Match a [`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use their respective (`onto`)[`Onto::onto`] method
    /// onto a [`Bounded`].
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
    fn onto(
        &self,
        item: &<BaseDom<'a, N, T,U> as Domain>::TypeDom,
        target: &Bounded<Out>,
    ) -> Result<Out, DomainError> {
        match self {
            Self::Bounded(d) => match item{
                Self::TypeDom::Bounded(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Unit(d) => match item{
                Self::TypeDom::Unit(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Bool(d) => match item{
                Self::TypeDom::Bool(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Cat(d) => match item{
                Self::TypeDom::Cat(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
        }
    }
}


impl<'a, const N: usize, T, U, Out> Onto<Unit<Out>> for BaseDom<'a, N, T, U>
where
    T : BoundedBounds,
    U : UnitBounds,
    Out: UnitBounds,
    f64: AsPrimitive<Out>
{
    /// [`Onto`] function between a [`BaseDom`] and a [`Unit`] [`Domain`].
    ///
    /// Match a [`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use their respective (`onto`)[`Onto::onto`] method
    /// onto a [`Unit`].
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
    fn onto(
        &self,
        item: &<BaseDom<'a, N, T,U> as Domain>::TypeDom,
        target: &Unit<Out>,
    ) -> Result<Out, DomainError> {
        match self {
            Self::Bounded(d) => match item{
                Self::TypeDom::Bounded(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Unit(d) => match item{
                Self::TypeDom::Unit(_i) => unreachable!("Converting a value from Unit onto Unit is not implemented, and it should not occur."),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Bool(d) => match item{
                Self::TypeDom::Bool(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Cat(d) => match item{
                Self::TypeDom::Cat(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
        }
    }
}

impl<'a, const N: usize,T,U> Onto<Bool> for BaseDom<'a, N, T, U>
where
    T : BoundedBounds,
    U : UnitBounds,
    f64: AsPrimitive<T>,
    f64: AsPrimitive<U>,
{
    /// [`Onto`] function between a [`BaseDom`] and a [`Bool`] [`Domain`].
    ///
    /// Match a [`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use their respective (`onto`)[`Onto::onto`] method
    /// onto a [`Bool`].
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Bool`]`<Out>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &<BaseDom<'a, N, T,U> as Domain>::TypeDom,
        target: &Bool,
    ) -> Result<<Bool as Domain>::TypeDom, DomainError> {
        match self {
            Self::Bounded(d) => match item{
                Self::TypeDom::Bounded(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Unit(d) => match item{
                Self::TypeDom::Unit(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Bool(d) => match item{
                Self::TypeDom::Bool(_i) => unreachable!("Converting a value from Bool onto Bool is not implemented, and it should not occur."),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Cat(d) => match item{
                Self::TypeDom::Cat(_i) => unreachable!("Converting a value from Cat onto Bool is not implemented, and it should not occur."),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
        }
    }
}


impl<'a, const N: usize, const M: usize,T,U> Onto<Cat<'a, N>> for BaseDom<'a, M, T, U>
where
    T : BoundedBounds,
    U : UnitBounds,
    f64: AsPrimitive<T>,
    f64: AsPrimitive<U>,
{
    /// [`Onto`] function between a [`BaseDom`] and a [`Cat`] [`Domain`].
    ///
    /// Match a [`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use their respective (`onto`)[`Onto::onto`] method
    /// onto a [`Cat`].
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Cat`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &<BaseDom<'a, M, T,U> as Domain>::TypeDom,
        target: &Cat<'a,N>,
    ) -> Result<<Cat<'a,N> as Domain>::TypeDom, DomainError> {
        match self {
            Self::Bounded(d) => match item{
                Self::TypeDom::Bounded(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Unit(d) => match item{
                Self::TypeDom::Unit(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Bool(d) => match item{
                Self::TypeDom::Bool(_i) => unreachable!("Converting a value from Bool onto Cat is not implemented, and it should not occur."),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Cat(d) => match item{
                Self::TypeDom::Cat(_i) => unreachable!("Converting a value from Cat onto Cat is not implemented, and it should not occur."),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
        }
    }
}


impl<'a, const N: usize, const M: usize,T,U,V,W> Onto<BaseDom<'a, N, V, W>> for BaseDom<'a, M, T, U>
where
    T : BoundedBounds,
    U : UnitBounds,
    V : BoundedBounds,
    W : UnitBounds,
    f64: AsPrimitive<T>,
    f64: AsPrimitive<U>,
    f64: AsPrimitive<V>,
    f64: AsPrimitive<W>,
{
    /// [`Onto`] function between a [`BaseDom`] and another [`BaseDom`] [`Domain`].
    ///
    /// Match a [`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use their respective (`onto`)[`Onto::onto`] method
    /// onto another [`BasedDom`].
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`BaseDom`]`<'a, M, T, U>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &<BaseDom<'a, M, T,U> as Domain>::TypeDom,
        target: &BaseDom<'a, N, V,W>,
    ) -> Result<<BaseDom<'a, N, V, W> as Domain>::TypeDom, DomainError> {
        match self {
            Self::Bounded(d) => match item{
                Self::TypeDom::Bounded(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Unit(d) => match item{
                Self::TypeDom::Unit(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Bool(d) => match item{
                Self::TypeDom::Bool(_i) => unreachable!("Converting a value from Bool onto Bool is not implemented, and it should not occur."),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Cat(d) => match item{
                Self::TypeDom::Cat(_i) => unreachable!("Converting a value from Cat onto Bool is not implemented, and it should not occur."),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
        }
    }
}
