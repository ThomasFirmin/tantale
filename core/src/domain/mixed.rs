use crate::{
    Cat, GridDom, GridInt, GridNat, GridReal, domain::{
        Domain, PreDomain,
        bool::Bool,
        bounded::{Int, Nat, Real},
        grid::GridBounds,
        onto::{Onto, OntoDom},
        unit::Unit,
    }, errors::OntoError,
    utils::aliases::Elem,
};

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

// -_-_-_-_-_-_-_-
// Grouped domains

/// A mixed [`Domain`], made of the 6 basic domains [`Real`], [`Nat`], [`Int`], [`Bool`], [`Cat`]
/// and [`Unit`]. And 4, discretized, grid versions of the basic domains: [`GridReal`], [`GridNat`], and [`GridInt`].
/// The [`TypeDom`](`Domain::TypeDom`) is a [`MixedTypeDom`].
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Mixed {
    Real(Real),
    Nat(Nat),
    Int(Int),
    Unit(Unit),
    Bool(Bool),
    Cat(Cat),
    GridReal(GridReal),
    GridNat(GridNat),
    GridInt(GridInt),
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
            Self::GridReal(d) => std::fmt::Display::fmt(&d, f),
            Self::GridNat(d) => std::fmt::Display::fmt(&d, f),
            Self::GridInt(d) => std::fmt::Display::fmt(&d, f),
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
            Self::GridReal(d) => std::fmt::Debug::fmt(&d, f),
            Self::GridNat(d) => std::fmt::Debug::fmt(&d, f),
            Self::GridInt(d) => std::fmt::Debug::fmt(&d, f),
        }
    }
}

/// Basic [`TypeDom`](`Domain::TypeDom`) of [`Mixed`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MixedTypeDom {
    Real(Elem<Real>),
    Nat(Elem<Nat>),
    Int(Elem<Int>),
    Bool(Elem<Bool>),
    Cat(Elem<Cat>),
    Unit(Elem<Unit>),
    GridReal(Elem<GridReal>),
    GridNat(Elem<GridNat>),
    GridInt(Elem<GridInt>),
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
            Self::GridReal(d) => std::fmt::Display::fmt(&d, f),
            Self::GridNat(d) => std::fmt::Display::fmt(&d, f),
            Self::GridInt(d) => std::fmt::Display::fmt(&d, f),
        }
    }
}

impl Default for MixedTypeDom {
    fn default() -> Self {
        MixedTypeDom::Real(Elem::<Real>::default())
    }
}
impl PreDomain for Mixed {}
impl Domain for Mixed {
    type Element = MixedTypeDom;

    /// Samples a point from the [`Mixed`] domain, according to each sub-domain sampling method.
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Element {
        match self {
            Self::Real(e) => MixedTypeDom::Real(e.sample(rng)),
            Self::Nat(e) => MixedTypeDom::Nat(e.sample(rng)),
            Self::Int(e) => MixedTypeDom::Int(e.sample(rng)),
            Self::Bool(e) => MixedTypeDom::Bool(e.sample(rng)),
            Self::Cat(e) => MixedTypeDom::Cat(e.sample(rng)),
            Self::Unit(e) => MixedTypeDom::Unit(e.sample(rng)),
            Self::GridReal(e) => MixedTypeDom::Real(e.sample(rng)),
            Self::GridNat(e) => MixedTypeDom::Nat(e.sample(rng)),
            Self::GridInt(e) => MixedTypeDom::Int(e.sample(rng)),
        }
    }

    /// Checks if a given point is in the [`Mixed`] domain, according to each sub-domain `contains` method.
    fn contains(&self, item: &Self::Element) -> bool {
        match (self, item) {
            (Self::Real(e), MixedTypeDom::Real(i)) => e.contains(i),
            (Self::Nat(e), MixedTypeDom::Nat(i)) => e.contains(i),
            (Self::Int(e), MixedTypeDom::Int(i)) => e.contains(i),
            (Self::Bool(e), MixedTypeDom::Bool(i)) => e.contains(i),
            (Self::Cat(e), MixedTypeDom::Cat(i)) => e.contains(i),
            (Self::Unit(e), MixedTypeDom::Unit(i)) => e.contains(i),
            (Self::GridReal(e), MixedTypeDom::Real(i)) => e.contains(i),
            (Self::GridNat(e), MixedTypeDom::Nat(i)) => e.contains(i),
            (Self::GridInt(e), MixedTypeDom::Int(i)) => e.contains(i),
            _ => false, // Type mismatch
        }
    }
}

impl Onto<Real> for Mixed {
    type TargetItem = Elem<Real>;
    type Item = Elem<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and a [`Real`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`](crate::Bounded), [`Bool`], [`Cat`] and [`Unit`].
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
            (Self::GridReal(d), MixedTypeDom::GridReal(i)) => d.onto(i, target),
            (Self::GridNat(d), MixedTypeDom::GridNat(i)) => d.onto(i, target),
            (Self::GridInt(d), MixedTypeDom::GridInt(i)) => d.onto(i, target),
            _ => Err(OntoError(format!(
                "Converting the value {:?} from {:?} onto Real is not implemented, and it should not occur.",
                item, self
            ))),
        }
    }
}
impl OntoDom<Real> for Mixed {}

impl Onto<Nat> for Mixed {
    type TargetItem = Elem<Nat>;
    type Item = Elem<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and a [`Nat`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`](crate::Bounded), [`Bool`], [`Cat`] and [`Unit`].
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
            (Self::GridReal(d), MixedTypeDom::GridReal(i)) => d.onto(i, target),
            (Self::GridNat(d), MixedTypeDom::GridNat(i)) => d.onto(i, target),
            (Self::GridInt(d), MixedTypeDom::GridInt(i)) => d.onto(i, target),
            _ => Err(OntoError(format!(
                "Converting the value {:?} from {:?} onto Nat is not implemented, and it should not occur.",
                item, self
            ))),
        }
    }
}
impl OntoDom<Nat> for Mixed {}

impl Onto<Int> for Mixed {
    type TargetItem = Elem<Int>;
    type Item = Elem<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and a [`Int`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`](crate::Bounded), [`Bool`], [`Cat`] and [`Unit`].
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
            (Self::GridReal(d), MixedTypeDom::GridReal(i)) => d.onto(i, target),
            (Self::GridNat(d), MixedTypeDom::GridNat(i)) => d.onto(i, target),
            (Self::GridInt(d), MixedTypeDom::GridInt(i)) => d.onto(i, target),
            _ => Err(OntoError(format!(
                "Converting the value {:?} from {:?} onto Int is not implemented, and it should not occur.",
                item, self
            ))),
        }
    }
}
impl OntoDom<Int> for Mixed {}

impl Onto<Unit> for Mixed {
    type TargetItem = Elem<Unit>;
    type Item = Elem<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and a [`Unit`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`](crate::Bounded), [`Bool`], [`Cat`] and [`Unit`].
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
            (Self::Bool(d), MixedTypeDom::Bool(i)) => d.onto(i, target),
            (Self::Cat(d), MixedTypeDom::Cat(i)) => d.onto(i, target),
            (Self::GridReal(d), MixedTypeDom::GridReal(i)) => d.onto(i, target),
            (Self::GridNat(d), MixedTypeDom::GridNat(i)) => d.onto(i, target),
            (Self::GridInt(d), MixedTypeDom::GridInt(i)) => d.onto(i, target),
            _ => Err(OntoError(format!(
                "Converting the value {:?} from {:?} onto Unit is not implemented, and it should not occur.",
                item, self
            ))),
        }
    }
}
impl OntoDom<Unit> for Mixed {}

impl Onto<Bool> for Mixed {
    type TargetItem = Elem<Bool>;
    type Item = Elem<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and a [`Bool`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`](crate::Bounded), [`Bool`], [`Cat`] and [`Unit`].
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
            _ => Err(OntoError(format!(
                "Converting the value {:?} from {:?} onto Int is not implemented, and it should not occur.",
                item, self
            ))),
        }
    }
}
impl OntoDom<Bool> for Mixed {}

impl<Out: GridBounds> Onto<GridDom<Out>> for Mixed {
    type TargetItem = Elem<GridDom<Out>>;
    type Item = Elem<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and a [`GridDom`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`](crate::Bounded), [`Bool`], [`GridDom`] and [`Unit`].
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Mixed`].
    /// * `target` - A borrowed targetted [`GridDom`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Mixed`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`GridDom`] domain.
    fn onto(
        &self,
        item: &Self::Item,
        target: &GridDom<Out>,
    ) -> Result<Self::TargetItem, OntoError> {
        match (self, item) {
            (Self::Real(d), MixedTypeDom::Real(i)) => d.onto(i, target),
            (Self::Nat(d), MixedTypeDom::Nat(i)) => d.onto(i, target),
            (Self::Int(d), MixedTypeDom::Int(i)) => d.onto(i, target),
            (Self::Cat(d), MixedTypeDom::Cat(i)) => d.onto(i, target),
            (Self::Unit(d), MixedTypeDom::Unit(i)) => d.onto(i, target),
            (Self::GridReal(d), MixedTypeDom::Real(i)) => d.onto(i, target),
            (Self::GridNat(d), MixedTypeDom::Nat(i)) => d.onto(i, target),
            (Self::GridInt(d), MixedTypeDom::Int(i)) => d.onto(i, target),
            _ => Err(OntoError(format!(
                "Converting the value {:?} from {:?} onto Int is not implemented, and it should not occur.",
                item, self
            ))),
        }
    }
}
impl<Out: GridBounds> OntoDom<GridDom<Out>> for Mixed {}

impl Onto<Mixed> for Mixed {
    type TargetItem = Elem<Mixed>;
    type Item = Elem<Mixed>;
    /// [`Onto`] function between a [`Mixed`] and another [`Mixed`] [`Domain`].
    ///
    /// Use the respective [`onto`](`Onto::onto`) method for each matched [`Mixed`] [`Bounded`](crate::Bounded), [`Bool`], [`Cat`] and [`Unit`].
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
                (Self::Real(d), MixedTypeDom::Real(item)) => d.onto(item, target),
                (Self::Nat(d), MixedTypeDom::Nat(item)) => d.onto(item, target),
                (Self::Int(d), MixedTypeDom::Int(item)) => d.onto(item, target),
                (Self::Unit(d), MixedTypeDom::Unit(item)) => d.onto(item, target),
                (Self::Bool(d), MixedTypeDom::Bool(item)) => d.onto(item, target),
                (Self::Cat(d), MixedTypeDom::Cat(item)) => d.onto(item, target),
                (Self::GridReal(d), MixedTypeDom::GridReal(item)) => d.onto(item, target),
                (Self::GridNat(d), MixedTypeDom::GridNat(item)) => d.onto(item, target),
                (Self::GridInt(d), MixedTypeDom::GridInt(item)) => d.onto(item, target),
                _ => Err(OntoError(format!(
                    "Converting the value {:?} from {:?} onto Mixed is not implemented, and it should not occur.",
                    item, self
                ))),
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

impl From<GridReal> for Mixed {
    fn from(value: GridReal) -> Self {
        Mixed::GridReal(value)
    }
}

impl From<GridNat> for Mixed {
    fn from(value: GridNat) -> Self {
        Mixed::GridNat(value)
    }
}

impl From<GridInt> for Mixed {
    fn from(value: GridInt) -> Self {
        Mixed::GridInt(value)
    }
}

impl From<Unit> for Mixed {
    fn from(value: Unit) -> Self {
        Mixed::Unit(value)
    }
}