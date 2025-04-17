use crate::core::domain::Domain;
use crate::core::domain::bounded::{Bounded,DomainBounded};
use crate::core::domain::unit::Unit;
use crate::core::domain::sampler::uniform_bool;
use crate::core::domain::onto::Onto;
use crate::core::domain::errors_domain::{DomainError,DomainOoBError};

use std::fmt::{self, Debug, Display};
use num::cast::AsPrimitive;
use num::{Float, Num, NumCast};
use rand::rngs::ThreadRng;
use rand::distr::uniform::SampleUniform;

// _-_-_-_-_-_-__-_-_-_-_-_-_-_
// Booleans domain

/// Describes a boolean domain .
///
/// # Examples
///
/// ```
/// use tantale::core::{Bool,Domain};
/// let dom = Bool::new();
///
/// let mut rng = rand::rng();
/// let sampler = dom.default_sampler();
/// assert!(dom.is_in(&sampler(&dom, &mut rng)));
/// ```
#[derive(Clone, Copy)]
pub struct Bool;
impl Bool {
    /// Fabric for a [`Bool`].
    ///
    pub fn new() -> Bool {
        Bool {}
    }
}
impl Domain for Bool {
    type TypeDom = bool;

    /// Default sampler for [`Bool`].
    /// See [`uniform_bool`].
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        uniform_bool
    }
    /// Method to check if a given point is in the domain.
    ///
    /// # Attributes
    ///
    /// * `point` : `&`[`Self`]`::`[`TypeDom`](Domain::TypeDom) :
    /// a point of the same type as the type of the domain.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Bool, Domain};
    ///
    /// let mut rng = rand::rng();
    /// let dom = Bool::new();
    ///
    /// let sampler = dom.default_sampler();
    /// assert!(dom.is_in(&sampler(&dom, &mut rng)));
    /// ```
    ///
    fn is_in(&self, _point: &Self::TypeDom) -> bool {
        true
    }
}
impl fmt::Display for Bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{T,F}}")
    }
}
impl fmt::Debug for Bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{T,F}}")
    }
}

impl<Out> Onto<Bounded<Out>> for Bool
where
    Out: Num
        + NumCast
        + PartialEq
        + PartialOrd
        + Clone
        + SampleUniform
        + AsPrimitive<f64>
        + Display
        + Debug,
    f64: AsPrimitive<Out>,
{
    /// [`Onto`] function between a [`Bool`] and a [`Bounded`] [`Domain`].
    ///
    /// Considering $l_{out}$ and $u_{out}$ the lower and upper bounds of
    /// the the output [`Bounded`] [`Domain`], and $x$ the item to be mapped.
    /// If `item` is `true` returns $u_{out}$, otherwise returns $l_{out}$ .
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
    fn onto(
        &self,
        item: &<Bool as Domain>::TypeDom,
        target: &Bounded<Out>,
    ) -> Result<Out, DomainError> {
        if self.is_in(item) {
            let mapped = if *item {
                target.upper()
            } else {
                target.lower()
            };
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


impl<Out> Onto<Unit<Out>> for Bool
where
    Out: Num
        + NumCast
        + Float
        + PartialEq
        + PartialOrd
        + Clone
        + SampleUniform
        + AsPrimitive<f64>
        + Display
        + Debug,
    f64: AsPrimitive<Out>,
{
    /// [`Onto`] function between a [`Bool`] and a [`Bounded`] [`Domain`].
    ///
    /// Considering $l_{out}$ and $u_{out}$ the lower and upper bounds of
    /// the the output [`Bounded`] [`Domain`], and $x$ the item to be mapped.
    /// If `item` is `true` returns $u_{out}$, otherwise returns $l_{out}$ .
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Bounded`]`<Out>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB(DomainOoBError`])
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &<Bool as Domain>::TypeDom,
        target: &Unit<Out>,
    ) -> Result<Out, DomainError> {
        if self.is_in(item) {
            let mapped = if *item {
                Out::from(1.0).unwrap()
            } else {
                Out::from(0.0).unwrap()
            };
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