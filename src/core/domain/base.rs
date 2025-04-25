use crate::core::domain::{
    Domain,
    bounded::{Real, Nat, Int},
    unit::Unit,
    bool::Bool,
    cat::Cat,
    errors_domain::{DomainError,DomainOoBError},
    onto::Onto,
};

use rand::prelude::ThreadRng;
use std::fmt::{Debug, Display};

// -_-_-_-_-_-_-_-
// Grouped domains

/// A [`enum`] [`BaseDom`] [`Domain`], made of the 6 basic domains [`Real`], [`Nat`], [`Int`], [`Bool`], [`Cat`]
/// and [`Unit`]. Used for mixed [`Element`] and [`Solution`].
/// The (`TypeDom`)[`Domain::TypeDom`] is an [`enum`] [`BaseTypeDom`].
///
#[derive(Clone,PartialEq)]
pub enum BaseDom<'a>
{
    Real(Real),
    Nat(Nat),
    Int(Int),
    Bool(Bool),
    Cat(Cat<'a>),
    Unit(Unit),
}

impl<'a> Display for BaseDom<'a>
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {

        match self {
            Self::Real(d) => std::fmt::Display::fmt(&d, f),
            Self::Nat(d) => std::fmt::Display::fmt(&d, f),
            Self::Int(d) => std::fmt::Display::fmt(&d, f),
            Self::Unit(d) => std::fmt::Display::fmt(&d, f),
            Self::Bool(d) => std::fmt::Display::fmt(&d, f),
            Self::Cat(d) => std::fmt::Display::fmt(&d, f),
        }
    }
}
impl<'a> Debug for BaseDom<'a>
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {

        match self {
            Self::Real(d) => std::fmt::Debug::fmt(&d, f),
            Self::Nat(d) => std::fmt::Debug::fmt(&d, f),
            Self::Int(d) => std::fmt::Debug::fmt(&d, f),
            Self::Unit(d) => std::fmt::Debug::fmt(&d, f),
            Self::Bool(d) => std::fmt::Debug::fmt(&d, f),
            Self::Cat(d) => std::fmt::Debug::fmt(&d, f),
        }
    }
}

/// Basic (`TypeDom`)[`Domain::TypeDom`] of [`BaseDom`].
/// 
#[derive(Debug,Copy,Clone,PartialEq)]
pub enum BaseTypeDom<'a>
{
    Real(<Real as Domain>::TypeDom),
    Nat(<Nat as Domain>::TypeDom),
    Int(<Int as Domain>::TypeDom),
    Bool(<Bool as Domain>::TypeDom),
    Cat(<Cat<'a> as Domain>::TypeDom),
    Unit(<Unit as Domain>::TypeDom),
}
impl<'a> Display for BaseTypeDom<'a>
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {

        match self {
            Self::Real(d) => std::fmt::Display::fmt(&d, f),
            Self::Nat(d) => std::fmt::Display::fmt(&d, f),
            Self::Int(d) => std::fmt::Display::fmt(&d, f),
            Self::Unit(d) => std::fmt::Display::fmt(&d, f),
            Self::Bool(d) => std::fmt::Display::fmt(&d, f),
            Self::Cat(d) => std::fmt::Display::fmt(&d, f),
        }
    }
}

impl <'a> Domain for BaseDom<'a>
{
    type TypeDom = BaseTypeDom<'a>;
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        
        |d,rng|  match d{
            Self::Real(e) => BaseTypeDom::Real(e.default_sampler()(e,rng)),
            Self::Nat(e) => BaseTypeDom::Nat(e.default_sampler()(e,rng)),
            Self::Int(e) => BaseTypeDom::Int(e.default_sampler()(e,rng)),
            Self::Bool(e) => BaseTypeDom::Bool(e.default_sampler()(e,rng)),
            Self::Cat(e) => BaseTypeDom::Cat(e.default_sampler()(e,rng)),
            Self::Unit(e) => BaseTypeDom::Unit(e.default_sampler()(e,rng)),
        }
    }

    fn is_in(&self, item: &Self::TypeDom) -> bool {
        match self {
            Self::Real(d) => match item{
                Self::TypeDom::Real(i) => d.is_in(i),
                _ => false,
            },
            Self::Nat(d) => match item{
                Self::TypeDom::Nat(i) => d.is_in(i),
                _ => false,
            },
            Self::Int(d) => match item{
                Self::TypeDom::Int(i) => d.is_in(i),
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

impl<'a> Onto<Real> for BaseDom<'a>
{
    /// [`Onto`] function between a [`BaseDom`] and a [`Real`] [`Domain`].
    ///
    /// Match a [`BaseDom`] [`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use their respective (`onto`)[`Onto::onto`] method
    /// onto a [`Real`].
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Real`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &<BaseDom<'a> as Domain>::TypeDom,
        target: &Real,
    ) -> Result<<Real as Domain>::TypeDom, DomainError> {
        match self {
            Self::Real(d) => match item{
                Self::TypeDom::Real(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Nat(d) => match item{
                Self::TypeDom::Nat(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Int(d) => match item{
                Self::TypeDom::Int(i) => d.onto(i,target),
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

impl<'a> Onto<Nat> for BaseDom<'a>
{
    /// [`Onto`] function between a [`BaseDom`] and a [`Nat`] [`Domain`].
    ///
    /// Match a [`BaseDom`] [`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use their respective (`onto`)[`Onto::onto`] method
    /// onto a [`Nat`].
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Nat`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &<BaseDom<'a> as Domain>::TypeDom,
        target: &Nat,
    ) -> Result<<Nat as Domain>::TypeDom, DomainError> {
        match self {
            Self::Real(d) => match item{
                Self::TypeDom::Real(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Nat(d) => match item{
                Self::TypeDom::Nat(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Int(d) => match item{
                Self::TypeDom::Int(i) => d.onto(i,target),
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

impl<'a> Onto<Int> for BaseDom<'a>
{
    /// [`Onto`] function between a [`BaseDom`] and a [`Int`] [`Domain`].
    ///
    /// Match a [`BaseDom`] [`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use their respective (`onto`)[`Onto::onto`] method
    /// onto a [`Int`].
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Int`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &<BaseDom<'a> as Domain>::TypeDom,
        target: &Int,
    ) -> Result<<Int as Domain>::TypeDom, DomainError> {
        match self {
            Self::Real(d) => match item{
                Self::TypeDom::Real(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Nat(d) => match item{
                Self::TypeDom::Nat(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Int(d) => match item{
                Self::TypeDom::Int(i) => d.onto(i,target),
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

impl<'a> Onto<Unit> for BaseDom<'a>
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
        item: &<BaseDom<'a> as Domain>::TypeDom,
        target: &Unit,
    ) -> Result<f64, DomainError> {
        match self {
            Self::Real(d) => match item{
                Self::TypeDom::Real(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Nat(d) => match item{
                Self::TypeDom::Nat(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Int(d) => match item{
                Self::TypeDom::Int(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Unit(_d) => unreachable!("Converting a value from Unit onto Unit is not implemented, and it should not occur."),
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

impl<'a> Onto<Bool> for BaseDom<'a>
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
        item: &<BaseDom<'a> as Domain>::TypeDom,
        target: &Bool,
    ) -> Result<<Bool as Domain>::TypeDom, DomainError> {
        match self {
            Self::Real(d) => match item{
                Self::TypeDom::Real(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Nat(d) => match item{
                Self::TypeDom::Nat(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Int(d) => match item{
                Self::TypeDom::Int(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Unit(d) => match item{
                Self::TypeDom::Unit(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Bool(_d) => unreachable!("Converting a value from Bool onto Bool is not implemented, and it should not occur."),
            Self::Cat(d) => match item{
                Self::TypeDom::Cat(_i) => unreachable!("Converting a value from Cat onto Bool is not implemented, and it should not occur."),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
        }
    }
}


impl<'a> Onto<Cat<'a>> for BaseDom<'a>
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
        item: &<BaseDom<'a> as Domain>::TypeDom,
        target: &Cat<'a>,
    ) -> Result<<Cat<'a> as Domain>::TypeDom, DomainError> {
        match self {
            Self::Real(d) => match item{
                Self::TypeDom::Real(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Nat(d) => match item{
                Self::TypeDom::Nat(i) => d.onto(i,target),
                _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
            },
            Self::Int(d) => match item{
                Self::TypeDom::Int(i) => d.onto(i,target),
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
            Self::Cat(_d) => unreachable!("Converting a value from Cat onto Cat is not implemented, and it should not occur."),
        }
    }
}


impl<'a> Onto<BaseDom<'a>> for BaseDom<'a>
{
    /// [`Onto`] function between a [`BaseDom`] and another [`BaseDom`] [`Domain`].
    ///
    /// Match a [`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use their respective (`onto`)[`Onto::onto`] method
    /// onto another [`BasedDom`].
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`BaseDom`]`<'a, M, T>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &<BaseDom<'a> as Domain>::TypeDom,
        target: &BaseDom<'a>,
    ) -> Result<<BaseDom<'a> as Domain>::TypeDom, DomainError> {
        if self==target{
            Ok(item.clone())
        }
        else{
            match self {
                Self::Real(d) => match item{
                    Self::TypeDom::Real(i) => d.onto(i,target),
                    _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
                },
                Self::Nat(d) => match item{
                    Self::TypeDom::Nat(i) => d.onto(i,target),
                    _ => Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, d)))),
                },
                Self::Int(d) => match item{
                    Self::TypeDom::Int(i) => d.onto(i,target),
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
}

impl <'a> From<Real> for BaseDom<'a>
{
    fn from(value: Real) -> Self {
        BaseDom::Real(value)
    }
}
impl <'a> From<Nat> for BaseDom<'a>
{
    fn from(value: Nat) -> Self {
        BaseDom::Nat(value)
    }
}
impl <'a> From<Int> for BaseDom<'a>
{
    fn from(value: Int) -> Self {
        BaseDom::Int(value)
    }
}
impl <'a> From<Bool> for BaseDom<'a>
{
    fn from(value: Bool) -> Self {
        BaseDom::Bool(value)
    }
}

impl <'a> From<Cat<'a>> for BaseDom<'a>
{
    fn from(value: Cat<'a>) -> Self {
        BaseDom::Cat(value)
    }
}

impl <'a> From<Unit> for BaseDom<'a>
{
    fn from(value: Unit) -> Self {
        BaseDom::Unit(value)
    }
}