use crate::domain::{
    bool::Bool,
    bounded::{Int, Nat, Real},
    cat::Cat,
    derrors::{DomainError, DomainOoBError},
    onto::{Onto, OntoOutput},
    unit::Unit,
    Domain, TypeDom,
};

use rand::prelude::ThreadRng;
use std::fmt::{Debug, Display};

// -_-_-_-_-_-_-_-
// Grouped domains

/// A [`enum`] [`BaseDom`] [`Domain`], made of the 6 basic domains [`Real`], [`Nat`], [`Int`], [`Bool`], [`Cat`]
/// and [`Unit`]. Used for mixed [`Element`] and [`Solution`].
/// The (`TypeDom`)[`Domain::TypeDom`] is an [`enum`] [`BaseTypeDom`].
///
#[derive(Clone, PartialEq)]
pub enum BaseDom {
    Real(Real),
    Nat(Nat),
    Int(Int),
    Bool(Bool),
    Cat(Cat),
    Unit(Unit),
}

impl Display for BaseDom {
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
impl Debug for BaseDom {
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
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BaseTypeDom {
    Real(TypeDom<Real>),
    Nat(TypeDom<Nat>),
    Int(TypeDom<Int>),
    Bool(TypeDom<Bool>),
    Cat(TypeDom<Cat>),
    Unit(TypeDom<Unit>),
}
impl Display for BaseTypeDom {
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
impl Default for BaseTypeDom {
    fn default() -> Self {
        BaseTypeDom::Real(TypeDom::<Real>::default())
    }
}

impl Domain for BaseDom {
    type TypeDom = BaseTypeDom;
    fn sample(&self, rng: &mut ThreadRng) -> Self::TypeDom {
        match self {
            Self::Real(e) => BaseTypeDom::Real(e.sample(rng)),
            Self::Nat(e) => BaseTypeDom::Nat(e.sample(rng)),
            Self::Int(e) => BaseTypeDom::Int(e.sample(rng)),
            Self::Bool(e) => BaseTypeDom::Bool(e.sample(rng)),
            Self::Cat(e) => BaseTypeDom::Cat(e.sample(rng)),
            Self::Unit(e) => BaseTypeDom::Unit(e.sample(rng)),
        }
    }

    fn is_in(&self, item: &Self::TypeDom) -> bool {
        match self {
            Self::Real(d) => match item {
                Self::TypeDom::Real(i) => d.is_in(i),
                _ => false,
            },
            Self::Nat(d) => match item {
                Self::TypeDom::Nat(i) => d.is_in(i),
                _ => false,
            },
            Self::Int(d) => match item {
                Self::TypeDom::Int(i) => d.is_in(i),
                _ => false,
            },
            Self::Unit(d) => match item {
                Self::TypeDom::Unit(i) => d.is_in(i),
                _ => false,
            },
            Self::Bool(d) => match item {
                Self::TypeDom::Bool(i) => d.is_in(i),
                _ => false,
            },
            Self::Cat(d) => match item {
                Self::TypeDom::Cat(i) => d.is_in(i),
                _ => false,
            },
        }
    }
}

impl Onto<Real> for BaseDom {
    /// [`Onto`] function between a [`BaseDom`] and a [`Real`] [`Domain`].
    ///
    /// Match a [`BaseDom`] [`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use their respective (`onto`)[`Onto::onto`] method
    /// onto a [`Real`].
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Real`] - A borrowed targetted [`Domain`].
    ///
    ///
    fn onto(&self, item: &TypeDom<BaseDom>, target: &Real) -> OntoOutput<Real> {
        match self {
            Self::Real(d) => match item {
                Self::TypeDom::Real(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Nat(d) => match item {
                Self::TypeDom::Nat(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Int(d) => match item {
                Self::TypeDom::Int(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Unit(d) => match item {
                Self::TypeDom::Unit(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Bool(d) => match item {
                Self::TypeDom::Bool(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Cat(d) => match item {
                Self::TypeDom::Cat(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
        }
    }
}

impl Onto<Nat> for BaseDom {
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
    fn onto(&self, item: &TypeDom<BaseDom>, target: &Nat) -> OntoOutput<Nat> {
        match self {
            Self::Real(d) => match item {
                Self::TypeDom::Real(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Nat(d) => match item {
                Self::TypeDom::Nat(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Int(d) => match item {
                Self::TypeDom::Int(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Unit(d) => match item {
                Self::TypeDom::Unit(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Bool(d) => match item {
                Self::TypeDom::Bool(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Cat(d) => match item {
                Self::TypeDom::Cat(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
        }
    }
}

impl Onto<Int> for BaseDom {
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
    fn onto(&self, item: &TypeDom<BaseDom>, target: &Int) -> OntoOutput<Int> {
        match self {
            Self::Real(d) => match item {
                Self::TypeDom::Real(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Nat(d) => match item {
                Self::TypeDom::Nat(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Int(d) => match item {
                Self::TypeDom::Int(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Unit(d) => match item {
                Self::TypeDom::Unit(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Bool(d) => match item {
                Self::TypeDom::Bool(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
            Self::Cat(d) => match item {
                Self::TypeDom::Cat(i) => d.onto(i, target),
                _ => Err(DomainError::OoB(DomainOoBError(format!(
                    "{} input not in {}",
                    item, d
                )))),
            },
        }
    }
}

impl Onto<Unit> for BaseDom {
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
    fn onto(&self, item: &TypeDom<BaseDom>, target: &Unit) -> OntoOutput<Unit> {
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

impl Onto<Bool> for BaseDom {
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
    fn onto(&self, item: &TypeDom<BaseDom>, target: &Bool) -> OntoOutput<Bool> {
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

impl Onto<Cat> for BaseDom {
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
    fn onto(&self, item: &TypeDom<BaseDom>, target: &Cat) -> OntoOutput<Cat> {
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

impl Onto<BaseDom> for BaseDom {
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
    fn onto(&self, item: &TypeDom<BaseDom>, target: &BaseDom) -> OntoOutput<BaseDom> {
        if self == target {
            Ok(*item)
        } else {
            match self {
                Self::Real(d) => match item {
                    Self::TypeDom::Real(i) => d.onto(i, target),
                    _ => Err(DomainError::OoB(DomainOoBError(format!(
                        "{} input not in {}",
                        item, d
                    )))),
                },
                Self::Nat(d) => match item {
                    Self::TypeDom::Nat(i) => d.onto(i, target),
                    _ => Err(DomainError::OoB(DomainOoBError(format!(
                        "{} input not in {}",
                        item, d
                    )))),
                },
                Self::Int(d) => match item {
                    Self::TypeDom::Int(i) => d.onto(i, target),
                    _ => Err(DomainError::OoB(DomainOoBError(format!(
                        "{} input not in {}",
                        item, d
                    )))),
                },
                Self::Unit(d) => match item {
                    Self::TypeDom::Unit(i) => d.onto(i, target),
                    _ => Err(DomainError::OoB(DomainOoBError(format!(
                        "{} input not in {}",
                        item, d
                    )))),
                },
                Self::Bool(d) => match item {
                    Self::TypeDom::Bool(i) => d.onto(i, target),
                    _ => Err(DomainError::OoB(DomainOoBError(format!(
                        "{} input not in {}",
                        item, d
                    )))),
                },
                Self::Cat(d) => match item {
                    Self::TypeDom::Cat(i) => d.onto(i, target),
                    _ => Err(DomainError::OoB(DomainOoBError(format!(
                        "{} input not in {}",
                        item, d
                    )))),
                },
            }
        }
    }
}

impl From<Real> for BaseDom {
    fn from(value: Real) -> Self {
        BaseDom::Real(value)
    }
}
impl From<Nat> for BaseDom {
    fn from(value: Nat) -> Self {
        BaseDom::Nat(value)
    }
}
impl From<Int> for BaseDom {
    fn from(value: Int) -> Self {
        BaseDom::Int(value)
    }
}
impl From<Bool> for BaseDom {
    fn from(value: Bool) -> Self {
        BaseDom::Bool(value)
    }
}

impl From<Cat> for BaseDom {
    fn from(value: Cat) -> Self {
        BaseDom::Cat(value)
    }
}

impl From<Unit> for BaseDom {
    fn from(value: Unit) -> Self {
        BaseDom::Unit(value)
    }
}
