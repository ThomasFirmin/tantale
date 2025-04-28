use crate::domain::{
    base::{BaseDom, BaseTypeDom},
    bounded::{Bounded, DomainBounded},
    derrors::{DomainError, DomainOoBError},
    onto::Onto,
    sampler::uniform_bool,
    unit::Unit,
    Domain,
};

use num::cast::AsPrimitive;
use num::{Num, NumCast};
use rand::{distr::uniform::SampleUniform, rngs::ThreadRng};
use std::fmt::{self, Debug, Display};

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

impl PartialEq for Bool {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Domain for Bool {
    type TypeDom = bool;

    /// Default sampler for [`Bool`].
    /// See [`uniform_bool`].
    fn sample(&self, rng: &mut ThreadRng) -> Self::TypeDom {
        uniform_bool(self, rng)
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

impl Onto<Unit> for Bool {
    /// [`Onto`] function between a [`Bool`] and a [`Bounded`] [`Domain`].
    ///
    /// Considering $l_{out}$ and $u_{out}$ the lower and upper bounds of
    /// the the output [`Bounded`] [`Domain`], and $x$ the item to be mapped.
    /// If `item` is `true` returns $u_{out}$, otherwise returns $l_{out}$ .
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Unit`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB(DomainOoBError`])
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &<Bool as Domain>::TypeDom, target: &Unit) -> Result<f64, DomainError> {
        if self.is_in(item) {
            let mapped = if *item { 1.0 } else { 0.0 };
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

impl Onto<BaseDom> for Bool {
    /// [`Onto`] function between a [`Bool`] [`Domain`] and a [`BaseDom`][`Domain`].
    ///
    //// Match a targetted[`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use (`onto`)[`Onto::onto`] of [`Self`].
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
    fn onto(
        &self,
        item: &bool,
        target: &BaseDom,
    ) -> Result<<BaseDom as Domain>::TypeDom, DomainError> {
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
            BaseDom::Unit(d) => match self.onto(item, d) {
                Ok(i) => Ok(BaseTypeDom::Unit(i)),
                Err(e) => Err(e),
            },
            BaseDom::Bool(_d) => unreachable!("Converting a value from Bool onto Bool is not implemented, and it should not occur."),
            BaseDom::Cat(_d) => unreachable!("Converting a value from Unit onto Unit is not implemented, and it should not occur."),
        }
    }
}

impl From<BaseDom> for Bool {
    fn from(value: BaseDom) -> Self {
        match value {
            BaseDom::Bool(d) => d,
            _ => unreachable!("Can only From<BaseDom> with Bool."),
        }
    }
}
