use crate::{
    domain::{
        Domain, PreDomain, TypeDom,
        bool::Bool,
        bounded::{Int, Nat, Real},
        cat::Cat,
        onto::{Onto, OntoDom},
        unit::Unit,
    },
    errors::OntoError,
    recorder::csv::CSVWritable,
};

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

// -_-_-_-_-_-_-_-
// Grouped domains

/// A mixed [`Domain`], made of the 6 basic domains [`Real`], [`Nat`], [`Int`], [`Bool`], [`Cat`]
/// and [`Unit`].
/// The [`TypeDom`](`Domain::TypeDom`) is a [`MixedTypeDom`].
#[derive(Clone, PartialEq)]
pub enum Mixed {
    Real(Real),
    Nat(Nat),
    Int(Int),
    Bool(Bool),
    Cat(Cat),
    Unit(Unit),
}

impl Display for Mixed {
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
impl Debug for Mixed {
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

/// Basic [`TypeDom`](`Domain::TypeDom`) of [`Mixed`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MixedTypeDom {
    Real(TypeDom<Real>),
    Nat(TypeDom<Nat>),
    Int(TypeDom<Int>),
    Bool(TypeDom<Bool>),
    Cat(TypeDom<Cat>),
    Unit(TypeDom<Unit>),
}
impl Display for MixedTypeDom {
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

impl Default for MixedTypeDom {
    fn default() -> Self {
        MixedTypeDom::Real(TypeDom::<Real>::default())
    }
}
impl PreDomain for Mixed {}
impl Domain for Mixed {
    type TypeDom = MixedTypeDom;

    /// Samples a point from the [`Mixed`] domain, according to each sub-domain sampling method.
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::TypeDom {
        match self {
            Self::Real(e) => MixedTypeDom::Real(e.sample(rng)),
            Self::Nat(e) => MixedTypeDom::Nat(e.sample(rng)),
            Self::Int(e) => MixedTypeDom::Int(e.sample(rng)),
            Self::Bool(e) => MixedTypeDom::Bool(e.sample(rng)),
            Self::Cat(e) => MixedTypeDom::Cat(e.sample(rng)),
            Self::Unit(e) => MixedTypeDom::Unit(e.sample(rng)),
        }
    }

    /// Checks if a given point is in the [`Mixed`] domain, according to each sub-domain `is_in` method.
    fn is_in(&self, item: &Self::TypeDom) -> bool {
        match (self, item) {
            (Self::Real(e), MixedTypeDom::Real(i)) => e.is_in(i),
            (Self::Nat(e), MixedTypeDom::Nat(i)) => e.is_in(i),
            (Self::Int(e), MixedTypeDom::Int(i)) => e.is_in(i),
            (Self::Bool(e), MixedTypeDom::Bool(i)) => e.is_in(i),
            (Self::Cat(e), MixedTypeDom::Cat(i)) => e.is_in(i),
            (Self::Unit(e), MixedTypeDom::Unit(i)) => e.is_in(i),
            _ => false, // Type mismatch
        }
    }
}

impl Onto<Real> for Mixed {
    type TargetItem = TypeDom<Real>;
    type Item = TypeDom<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and a [`Real`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`], [`Bool`], [`Cat`] and [`Unit`].
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Mixed`].
    /// * `target` - A borrowed targetted [`Real`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Mixed`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Real`] domain.
    fn onto(&self, item: &Self::Item, target: &Real) -> Result<Self::TargetItem, OntoError> {
        match (self, item) {
            (Self::Real(d), MixedTypeDom::Real(i)) => d.onto(i, target),
            (Self::Nat(d), MixedTypeDom::Nat(i)) => d.onto(i, target),
            (Self::Int(d), MixedTypeDom::Int(i)) => d.onto(i, target),
            (Self::Unit(d), MixedTypeDom::Unit(i)) => d.onto(i, target),
            (Self::Bool(d), MixedTypeDom::Bool(i)) => d.onto(i, target),
            (Self::Cat(d), MixedTypeDom::Cat(i)) => d.onto(i, target),
            _ => Err(OntoError(format!("{} input not in {}", item, self))),
        }
    }
}
impl OntoDom<Real> for Mixed {}

impl Onto<Nat> for Mixed {
    type TargetItem = TypeDom<Nat>;
    type Item = TypeDom<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and a [`Nat`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`], [`Bool`], [`Cat`] and [`Unit`].
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Mixed`].
    /// * `target` - A borrowed targetted [`Nat`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Mixed`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Nat`] domain.
    fn onto(&self, item: &Self::Item, target: &Nat) -> Result<Self::TargetItem, OntoError> {
        match (self, item) {
            (Self::Real(d), MixedTypeDom::Real(i)) => d.onto(i, target),
            (Self::Nat(d), MixedTypeDom::Nat(i)) => d.onto(i, target),
            (Self::Int(d), MixedTypeDom::Int(i)) => d.onto(i, target),
            (Self::Unit(d), MixedTypeDom::Unit(i)) => d.onto(i, target),
            (Self::Bool(d), MixedTypeDom::Bool(i)) => d.onto(i, target),
            (Self::Cat(d), MixedTypeDom::Cat(i)) => d.onto(i, target),
            _ => Err(OntoError(format!("{} input not in {}", item, self))),
        }
    }
}
impl OntoDom<Nat> for Mixed {}

impl Onto<Int> for Mixed {
    type TargetItem = TypeDom<Int>;
    type Item = TypeDom<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and a [`Int`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`], [`Bool`], [`Cat`] and [`Unit`].
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Mixed`].
    /// * `target` - A borrowed targetted [`Int`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Mixed`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Int`] domain.
    fn onto(&self, item: &Self::Item, target: &Int) -> Result<Self::TargetItem, OntoError> {
        match (self, item) {
            (Self::Real(d), MixedTypeDom::Real(i)) => d.onto(i, target),
            (Self::Nat(d), MixedTypeDom::Nat(i)) => d.onto(i, target),
            (Self::Int(d), MixedTypeDom::Int(i)) => d.onto(i, target),
            (Self::Unit(d), MixedTypeDom::Unit(i)) => d.onto(i, target),
            (Self::Bool(d), MixedTypeDom::Bool(i)) => d.onto(i, target),
            (Self::Cat(d), MixedTypeDom::Cat(i)) => d.onto(i, target),
            _ => Err(OntoError(format!("{} input not in {}", item, self))),
        }
    }
}
impl OntoDom<Int> for Mixed {}

impl Onto<Unit> for Mixed {
    type TargetItem = TypeDom<Unit>;
    type Item = TypeDom<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and a [`Unit`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`], [`Bool`], [`Cat`] and [`Unit`].
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Mixed`].
    /// * `target` - A borrowed targetted [`Unit`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Mixed`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Unit`] domain.
    fn onto(&self, item: &Self::Item, target: &Unit) -> Result<Self::TargetItem, OntoError> {
        match (self, item) {
            (Self::Real(d), MixedTypeDom::Real(i)) => d.onto(i, target),
            (Self::Nat(d), MixedTypeDom::Nat(i)) => d.onto(i, target),
            (Self::Int(d), MixedTypeDom::Int(i)) => d.onto(i, target),
            (Self::Unit(_), MixedTypeDom::Unit(_)) => unreachable!(
                "Converting a value from Unit onto Unit is not implemented, and it should not occur."
            ),
            (Self::Bool(d), MixedTypeDom::Bool(i)) => d.onto(i, target),
            (Self::Cat(d), MixedTypeDom::Cat(i)) => d.onto(i, target),
            _ => Err(OntoError(format!("{} input not in {}", item, self))),
        }
    }
}
impl OntoDom<Unit> for Mixed {}

impl Onto<Bool> for Mixed {
    type TargetItem = TypeDom<Bool>;
    type Item = TypeDom<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and a [`Bool`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`], [`Bool`], [`Cat`] and [`Unit`].
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Mixed`].
    /// * `target` - A borrowed targetted [`Bool`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Mixed`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Bool`] domain.
    fn onto(&self, item: &Self::Item, target: &Bool) -> Result<Self::TargetItem, OntoError> {
        match (self, item) {
            (Self::Real(d), MixedTypeDom::Real(i)) => d.onto(i, target),
            (Self::Nat(d), MixedTypeDom::Nat(i)) => d.onto(i, target),
            (Self::Int(d), MixedTypeDom::Int(i)) => d.onto(i, target),
            (Self::Unit(d), MixedTypeDom::Unit(i)) => d.onto(i, target),
            (Self::Bool(_), MixedTypeDom::Bool(_)) => unreachable!(
                "Converting a value from Bool onto Bool is not implemented, and it should not occur."
            ),
            (Self::Cat(_), MixedTypeDom::Cat(_)) => unreachable!(
                "Converting a value from Cat onto Bool is not implemented, and it should not occur."
            ),
            _ => Err(OntoError(format!("{} input not in {}", item, self))),
        }
    }
}
impl OntoDom<Bool> for Mixed {}

impl Onto<Cat> for Mixed {
    type TargetItem = TypeDom<Cat>;
    type Item = TypeDom<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and a [`Cat`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`], [`Bool`], [`Cat`] and [`Unit`].
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Mixed`].
    /// * `target` - A borrowed targetted [`Cat`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Mixed`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Cat`] domain.
    fn onto(&self, item: &Self::Item, target: &Cat) -> Result<Self::TargetItem, OntoError> {
        match self {
            Self::Real(d) => match item {
                MixedTypeDom::Real(i) => d.onto(i, target),
                _ => Err(OntoError(format!("{} input not in {}", item, d))),
            },
            Self::Nat(d) => match item {
                MixedTypeDom::Nat(i) => d.onto(i, target),
                _ => Err(OntoError(format!("{} input not in {}", item, d))),
            },
            Self::Int(d) => match item {
                MixedTypeDom::Int(i) => d.onto(i, target),
                _ => Err(OntoError(format!("{} input not in {}", item, d))),
            },
            Self::Unit(d) => match item {
                MixedTypeDom::Unit(i) => d.onto(i, target),
                _ => Err(OntoError(format!("{} input not in {}", item, d))),
            },
            Self::Bool(d) => match item {
                MixedTypeDom::Bool(_i) => unreachable!(
                    "Converting a value from Bool onto Cat is not implemented, and it should not occur."
                ),
                _ => Err(OntoError(format!("{} input not in {}", item, d))),
            },
            Self::Cat(_d) => unreachable!(
                "Converting a value from Cat onto Cat is not implemented, and it should not occur."
            ),
        }
    }
}
impl OntoDom<Cat> for Mixed {}

impl Onto<Mixed> for Mixed {
    type TargetItem = TypeDom<Mixed>;
    type Item = TypeDom<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and another [`Mixed`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`], [`Bool`], [`Cat`] and [`Unit`].
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Mixed`].
    /// * `target` - A borrowed targetted [`Mixed`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Mixed`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Mixed`] domain.
    fn onto(&self, item: &Self::Item, target: &Mixed) -> Result<Self::TargetItem, OntoError> {
        if self == target {
            Ok(item.clone())
        } else {
            match (self, item) {
                (Self::Real(d), MixedTypeDom::Real(i)) => d.onto(i, target),
                (Self::Nat(d), MixedTypeDom::Nat(i)) => d.onto(i, target),
                (Self::Int(d), MixedTypeDom::Int(i)) => d.onto(i, target),
                (Self::Unit(d), MixedTypeDom::Unit(i)) => d.onto(i, target),
                (Self::Bool(d), MixedTypeDom::Bool(i)) => d.onto(i, target),
                (Self::Cat(d), MixedTypeDom::Cat(i)) => d.onto(i, target),
                _ => Err(OntoError(format!("{} input not in {}", item, self))),
            }
        }
    }
}
impl OntoDom<Mixed> for Mixed {}

impl From<Real> for Mixed {
    fn from(value: Real) -> Self {
        Mixed::Real(value)
    }
}
impl From<Nat> for Mixed {
    fn from(value: Nat) -> Self {
        Mixed::Nat(value)
    }
}
impl From<Int> for Mixed {
    fn from(value: Int) -> Self {
        Mixed::Int(value)
    }
}
impl From<Bool> for Mixed {
    fn from(value: Bool) -> Self {
        Mixed::Bool(value)
    }
}

impl From<Cat> for Mixed {
    fn from(value: Cat) -> Self {
        Mixed::Cat(value)
    }
}

impl From<Unit> for Mixed {
    fn from(value: Unit) -> Self {
        Mixed::Unit(value)
    }
}

impl CSVWritable<(), MixedTypeDom> for Mixed {
    fn header(_elem: &()) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, comp: &MixedTypeDom) -> Vec<String> {
        match comp {
            MixedTypeDom::Real(s) => Vec::from([s.to_string()]),
            MixedTypeDom::Nat(s) => Vec::from([s.to_string()]),
            MixedTypeDom::Int(s) => Vec::from([s.to_string()]),
            MixedTypeDom::Bool(s) => Vec::from([s.to_string()]),
            MixedTypeDom::Cat(s) => Vec::from([s.to_string()]),
            MixedTypeDom::Unit(s) => Vec::from([s.to_string()]),
        }
    }
}
