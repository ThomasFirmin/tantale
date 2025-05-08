use crate::domain::{
    base::{BaseDom, BaseTypeDom},
    bounded::{Bounded, BoundedBounds, DomainBounded},
    derrors::{DomainError, DomainOoBError},
    onto::Onto,
    sampler::uniform_cat,
    unit::Unit,
    Domain,
};

use num::cast::AsPrimitive;
use rand::prelude::ThreadRng;
use std::fmt;

// _-_-_-_-_-_-__-_-_-_-_-_-_-_
// Categorical domain

/// Describes a non-ordinal categorical domain. It is made of features,
/// described by the private attribute `values`, an [`array`] of strings.
/// Each elements describes a unique feature.
/// The values can be accessed by the corresponding getter `values()`.
///
/// # Attributes
///
///  * `values` : `[&'a str; N]` - An array of the features defining the categorical [`Domain`].
///
/// # Examples
///
/// ```
/// use tantale::core::{Cat,Domain};
///
/// let mut rng = rand::rng();
/// let activation = ["relu", "tanh", "sigmoid"];
/// let check = ["relu", "tanh", "sigmoid"];
/// let dom = Cat::new(&activation);
///
/// let sampler = dom.default_sampler();
/// assert!(dom.is_in(&sampler(&dom, &mut rng)));
/// assert_eq!(dom.values(), &check);
/// ```
#[derive(Clone, Copy)]
pub struct Cat {
    pub values: &'static [&'static str],
}
impl Cat {
    /// Fabric for a [`Cat`].
    ///
    /// # Attributes
    ///
    ///  * `values` : `[&'a str; N]` - An array of the features defining the categorical [`Domain`].
    ///
    pub fn new(values: &'static [&'static str]) -> Cat {
        Cat { values }
    }
    /// Getter for values
    pub fn values(&self) -> &'static [&'static str] {
        self.values
    }
}

impl<'a> PartialEq for Cat {
    fn eq(&self, other: &Self) -> bool {
        self.values() == other.values()
    }
}

impl<'a> Domain for Cat {
    /// The type of a point within the domain is a `&'a str`, i.e. a pointer to a `str` from the `values`.
    type TypeDom = &'static str;

    /// Default sampler for [`Cat`] is a uniform choice within the `values`
    /// See [`uniform_cat`].
    fn sample(&self, rng: &mut ThreadRng) -> Self::TypeDom {
        uniform_cat(self, rng)
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
    /// use tantale::core::{Cat,Domain};
    ///
    /// let mut rng = rand::rng();
    /// let activation = ["relu", "tanh", "sigmoid"];
    /// let cat_1 = Cat::new(&activation);
    ///
    /// let sampler = cat_1.default_sampler();
    /// assert!(cat_1.is_in(&sampler(&cat_1, &mut rng)));
    ///
    fn is_in(&self, point: &Self::TypeDom) -> bool {
        self.values.contains(point)
    }
}

impl<'a> fmt::Display for Cat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let vsize = self.values.len() - 1;
        write!(f, "{{")?;
        for elem in self.values[..vsize].iter() {
            write!(f, "{}, ", elem)?;
        }
        write!(f, "{}}}", self.values[vsize])
    }
}

impl<'a> fmt::Debug for Cat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let vsize = self.values.len() - 1;
        write!(f, "{{")?;
        for elem in self.values[..vsize].iter() {
            write!(f, "{}, ", elem)?;
        }
        write!(f, "{}}}", self.values[vsize])
    }
}

impl<'a, Out> Onto<Bounded<Out>> for Cat
where
    Out: BoundedBounds,
    f64: AsPrimitive<Out>,
{
    /// [`Onto`] function between a [`Cat`] and a [`Bounded`] [`Domain`].
    ///
    /// Considering $l_{out}$ and $u_{out}$ the lower and upper bounds of
    /// the the output [`Bounded`] [`Domain`], $i$ the index of the item within `values` of [`Cat`],
    /// and $\ell_{en}$ the length of `values` of the [`Cat`] [`Domain`].
    ///
    /// The mapping is given by :
    ///
    /// $$ \frac{i}{\ell_{en}-1} \time (u_{out}-l_{out}) + l_{out} $$
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
        item: &<Cat as Domain>::TypeDom,
        target: &Bounded<Out>,
    ) -> Result<Out, DomainError> {
        let idx = self.values().iter().position(|n| n == item);

        match idx {
            Some(i) => {
                let a: f64 = (i + 1).as_();
                let b: f64 = self.values().len().as_();
                let c: f64 = target.width().as_();
                let mapped: Out = (a / b * c).as_() + target.lower();

                if target.is_in(&mapped) {
                    Ok(mapped)
                } else {
                    Err(DomainError::OoB(DomainOoBError(format!(
                        "{} -> {} mapped input not in {}",
                        item, mapped, target
                    ))))
                }
            }
            None => Err(DomainError::OoB(DomainOoBError(format!(
                "{} input not in {}",
                item, self
            )))),
        }
    }
}

impl<'a> Onto<Unit> for Cat {
    /// [`Onto`] function between a [`Cat`] and a [`Unit`] [`Domain`].
    ///
    /// Considering $i$ the index of the item within `values` of [`Cat`]
    /// and $\ell_{en}$ the length of `values` of the [`Cat`] [`Domain`].
    /// The mapping is given by :
    ///
    /// $$ \frac{i}{\ell_{en}-1} $$
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Unit`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &<Cat as Domain>::TypeDom, target: &Unit) -> Result<f64, DomainError> {
        let idx = self.values().iter().position(|n| n == item);

        match idx {
            Some(i) => {
                let a: f64 = (i + 1).as_();
                let b: f64 = self.values().len().as_();
                let mapped: f64 = a / b;

                if target.is_in(&mapped) {
                    Ok(mapped)
                } else {
                    Err(DomainError::OoB(DomainOoBError(format!(
                        "{} -> {} mapped input not in {}",
                        item, mapped, target
                    ))))
                }
            }
            None => Err(DomainError::OoB(DomainOoBError(format!(
                "{} input not in {}",
                item, self
            )))),
        }
    }
}

impl Onto<BaseDom> for Cat {
    /// [`Onto`] function between a [`Cat`] [`Domain`] and a [`BaseDom`][`Domain`].
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
        item: &<Cat as Domain>::TypeDom,
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
            BaseDom::Bool(_d) => unreachable!("Converting a value from Unit onto Unit is not implemented, and it should not occur."),
            BaseDom::Cat(_d) => unreachable!("Converting a value from Cat onto Cat is not implemented, and it should not occur."),
        }
    }
}

impl From<BaseDom> for Cat {
    fn from(value: BaseDom) -> Self {
        match value {
            BaseDom::Cat(d) => d,
            _ => unreachable!("Can only From<BaseDom> with Cat."),
        }
    }
}
