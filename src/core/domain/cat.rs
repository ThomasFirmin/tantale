use crate::core::domain::Domain;
use crate::core::domain::bounded::{Bounded,DomainBounded,BoundedBounds};
use crate::core::domain::unit::{Unit,UnitBounds};
use crate::core::domain::base::{BaseDom,BaseTypeDom};
use crate::core::domain::sampler::uniform_cat;
use crate::core::domain::onto::Onto;
use crate::core::domain::errors_domain::{DomainError,DomainOoBError};

use std::fmt::{self, Debug, Display};
use num::cast::AsPrimitive;
use num::{Float, Num, NumCast};
use rand::rngs::ThreadRng;
use rand::distr::uniform::SampleUniform;

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
pub struct Cat<'a, const N: usize> {
    values: &'a [&'a str; N],
}
impl<'a, const N: usize> Cat<'a, N> {
    /// Fabric for a [`Cat`].
    ///
    /// # Attributes
    ///
    ///  * `values` : `[&'a str; N]` - An array of the features defining the categorical [`Domain`].
    ///
    pub fn new(values: &'a [&'a str; N]) -> Cat<'a, N> {
        Cat { values }
    }
    /// Getter for values
    pub fn values(&self) -> &'a [&'a str; N] {
        self.values
    }
}
impl<'a, const N: usize> Domain for Cat<'a, N> {
    /// The type of a point within the domain is a `&'a str`, i.e. a pointer to a `str` from the `values`.
    type TypeDom = &'a str;

    /// Default sampler for [`Cat`] is a uniform choice within the `values`
    /// See [`uniform_cat`].
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        uniform_cat
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

impl<'a, const N: usize> fmt::Display for Cat<'a, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let vsize = self.values.len() - 1;
        write!(f, "{{")?;
        for elem in self.values[..vsize].iter() {
            write!(f, "{}, ", elem)?;
        }
        write!(f, "{}}}", self.values[vsize])
    }
}

impl<'a, const N: usize> fmt::Debug for Cat<'a, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let vsize = self.values.len() - 1;
        write!(f, "{{")?;
        for elem in self.values[..vsize].iter() {
            write!(f, "{}, ", elem)?;
        }
        write!(f, "{}}}", self.values[vsize])
    }
}


impl<'a, const N: usize, Out> Onto<Bounded<Out>> for Cat<'a, N>
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
        item: &<Cat<'a, N> as Domain>::TypeDom,
        target: &Bounded<Out>,
    ) -> Result<Out, DomainError> {
        let idx = self.values().iter().position(|n| n == item);

        match idx {
            Some(i) => {
                let a: f64 = (i).as_();
                let b: f64 = (self.values().len() - 1).as_();
                let c: f64 = target.width().as_();
                let mapped: Out = (a / b * c).as_() + target.lower();

                if target.is_in(&mapped) {
                    Ok(mapped)
                } else {
                    Err(DomainError::OoB(DomainOoBError(format!("{} -> {} mapped input not in {}", item, mapped, target))))
                }
            }
            None => {
                Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, self))))
            }
        }
    }
}

impl<'a, const N: usize, Out> Onto<Unit<Out>> for Cat<'a, N>
where
    Out : Num 
        + NumCast
        + Float
        + PartialEq
        + PartialOrd
        + SampleUniform
        + AsPrimitive<f64>
        + Display
        + Debug,
    f64: AsPrimitive<Out>,
{
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
    /// * `target` : `&`[`Unit`]`<Out>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &<Cat<'a, N> as Domain>::TypeDom, target: &Unit<Out>) -> Result<Out, DomainError> {
        let idx = self.values().iter().position(|n| n == item);

        match idx {
            Some(i) => {
                let a: f64 = (i).as_();
                let b: f64 = (self.values().len() - 1).as_();
                let mapped: Out = (a / b).as_();

                if target.is_in(&mapped) {
                    Ok(mapped)
                } else {
                    Err(DomainError::OoB(DomainOoBError(format!("{} -> {} mapped input not in {}", item, mapped, target))))
                }
            }
            None => {
                Err(DomainError::OoB(DomainOoBError(format!("{} input not in {}", item, self))))
            }
        }
    }
}

impl<'a, const N :usize, const M :usize,T,U> Onto<BaseDom<'a,N,T,U>> for Cat<'a,M>
where
    T:BoundedBounds,
    U:UnitBounds,

    f64: AsPrimitive<T>,
    f64: AsPrimitive<U>,
{
    /// [`Onto`] function between a [`Cat`] [`Domain`] and a [`BaseDom`][`Domain`].
    ///
    //// Match a targetted[`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use (`onto`)[`Onto::onto`] of [`Self`].
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
    fn onto(&self, item: &<Cat<'a,M> as Domain>::TypeDom, target: &BaseDom<'a,N,T,U>) -> Result<<BaseDom<'a,N,T,U> as Domain>::TypeDom, DomainError> {
        match target{
            BaseDom::Bounded(d) => {
                match self.onto(item, d) {
                    Ok(i) => Ok(BaseTypeDom::<'a,N,T,U>::Bounded(i)),
                    Err(e) => Err(e),
                }
            },
            BaseDom::Unit(d) => match self.onto(item, d) {
                Ok(i) => Ok(BaseTypeDom::<'a,N,T,U>::Unit(i)),
                Err(e) => Err(e),
            },
            BaseDom::Bool(_d) => unreachable!("Converting a value from Unit onto Unit is not implemented, and it should not occur."),
            BaseDom::Cat(_d) => unreachable!("Converting a value from Unit onto Unit is not implemented, and it should not occur."),
        }
    }
}