use crate::{
    domain::{
        Domain, PreDomain, TypeDom,
        bounded::{Bounded, BoundedBounds},
        mixed::{Mixed, MixedTypeDom},
        onto::{Onto, OntoDom},
        unit::Unit,
    },
    errors::OntoError,
    recorder::csv::CSVWritable,
    sampler::{CatDistribution, Sampler},
};

use num::cast::AsPrimitive;
use rand::prelude::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Describes a non-ordinal categorical domain. It is made of features,
/// described by a [`Vec`] of [`String`].
/// Each elements describes a unique feature.
///
/// # Attributes
///
///  * `values` - The features defining the categorical [`Domain`].
///  * `sampler` - The sampling algorithm used to sample within the categorical [`Domain`].
/// ```
#[derive(Clone)]
pub struct Cat {
    pub values: Vec<String>,
    pub sampler: CatDistribution,
}
impl Cat {
    /// Fabric for a [`Cat`].
    ///
    /// # Attributes
    ///
    /// * `values` - An iterator over `&str` describing the features of the categorical [`Domain`].
    /// * `sampler` - A sampler implementing the [`Sampler`] trait for [`Cat`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Cat,Domain,Uniform};
    ///
    /// let mut rng = rand::rng();
    /// let dom = Cat::new(["relu", "tanh", "sigmoid"], Uniform);
    ///
    /// let sample = dom.sample(&mut rng);
    /// assert!(dom.is_in(&sample));
    /// let check = ["relu", "tanh", "sigmoid"];
    /// assert_eq!(dom.values, check);
    /// ```
    pub fn new<'a, S: Sampler<Self> + Into<CatDistribution>, I: IntoIterator<Item = &'a str>>(
        values: I,
        sampler: S,
    ) -> Cat {
        Cat {
            values: values.into_iter().map(String::from).collect(),
            sampler: sampler.into(),
        }
    }
}

impl PartialEq for Cat {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl PreDomain for Cat {}
impl Domain for Cat {
    type TypeDom = String;

    /// Sample a `String` using the inner [`CatDistribution`] of [`Cat`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Cat,Domain,Uniform};
    ///
    /// let mut rng = rand::rng();
    /// let dom = Cat::new(["relu", "tanh", "sigmoid"], Uniform);
    ///
    /// let sample = dom.sample(&mut rng);
    /// assert!(dom.is_in(&sample));
    /// let check = ["relu", "tanh", "sigmoid"];
    /// assert_eq!(dom.values, check);
    /// ```
    fn sample<R: Rng>(&self, rng: &mut R) -> TypeDom<Self> {
        self.sampler.sample(self, rng)
    }

    /// Method to check if a given point is in the domain.
    ///
    /// # Attributes
    ///
    /// * `point` - A borrowed sample from a [`Cat`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Cat,Domain,Uniform};
    ///
    /// let mut rng = rand::rng();
    /// let dom = Cat::new(["relu", "tanh", "sigmoid"], Uniform);
    ///
    /// let sample = dom.sample(&mut rng);
    /// assert!(dom.is_in(&sample));
    /// let check = ["relu", "tanh", "sigmoid"];
    /// assert_eq!(dom.values, check);
    /// ```
    fn is_in(&self, point: &TypeDom<Self>) -> bool {
        self.values.contains(point)
    }
}

impl fmt::Display for Cat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let vsize = self.values.len() - 1;
        write!(f, "{{")?;
        for elem in self.values[..vsize].iter() {
            write!(f, "{}, ", elem)?;
        }
        write!(f, "{}}}", self.values[vsize])
    }
}

impl fmt::Debug for Cat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let vsize = self.values.len() - 1;
        write!(f, "{{")?;
        for elem in self.values[..vsize].iter() {
            write!(f, "{}, ", elem)?;
        }
        write!(f, "{}}}", self.values[vsize])
    }
}

impl<Out> Onto<Bounded<Out>> for Cat
where
    Out: BoundedBounds + Serialize + for<'a> Deserialize<'a>,
    f64: AsPrimitive<Out>,
{
    type Item = TypeDom<Cat>;
    type TargetItem = TypeDom<Bounded<Out>>;
    /// [`Onto`] function between a [`Cat`] and a [`Bounded`] [`Domain`].
    ///
    /// Considering $l_{out}$ and $u_{out}$ the lower and upper bounds of
    /// the the output [`Bounded`] [`Domain`], $i$ the index of the item within `values` of [`Cat`],
    /// and $\ell_{en}$ the length of `values` of the [`Cat`] [`Domain`].
    ///
    /// The mapping is given by :
    ///
    /// $$ \frac{i+1}{\ell_{en}} \time (u_{out}-l_{out}) + l_{out} $$
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &Self::Item,
        target: &Bounded<Out>,
    ) -> Result<Self::TargetItem, OntoError> {
        let idx = self.values.iter().position(|n| n == item);

        match idx {
            Some(i) => {
                let a: f64 = (i + 1).as_();
                let b: f64 = self.values.len().as_();
                let c: f64 = target.width.as_();
                let mapped: Out = (a / b * c).as_() + *target.bounds.start();

                if target.is_in(&mapped) {
                    Ok(mapped)
                } else {
                    Err(OntoError(format!("{} input not in {}", item, self)))
                }
            }
            None => Err(OntoError(format!("{} input not in {}", item, self))),
        }
    }
}
impl<Out> OntoDom<Bounded<Out>> for Cat
where
    Out: BoundedBounds + Serialize + for<'a> Deserialize<'a>,
    f64: AsPrimitive<Out>,
{
}

impl Onto<Unit> for Cat {
    type Item = TypeDom<Cat>;
    type TargetItem = TypeDom<Unit>;
    /// [`Onto`] function between a [`Cat`] and a [`Unit`] [`Domain`].
    ///
    /// Considering $i$ the index of the item within `values` of [`Cat`]
    /// and $\ell_{en}$ the length of `values` of the [`Cat`] [`Domain`].
    /// The mapping is given by :
    ///
    /// $$ \frac{i+1}{\ell_{en}} $$
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &Self::Item, target: &Unit) -> Result<Self::TargetItem, OntoError> {
        let idx = self.values.iter().position(|n| n == item);

        match idx {
            Some(i) => {
                let a: f64 = (i + 1).as_();
                let b: f64 = self.values.len().as_();
                let mapped: f64 = a / b;

                if target.is_in(&mapped) {
                    Ok(mapped)
                } else {
                    Err(OntoError(format!("{} input not in {}", item, self)))
                }
            }
            None => Err(OntoError(format!("{} input not in {}", item, self))),
        }
    }
}
impl OntoDom<Unit> for Cat {}

impl Onto<Mixed> for Cat {
    type Item = TypeDom<Cat>;
    type TargetItem = TypeDom<Mixed>;
    /// [`Onto`] function between a [`Cat`] [`Domain`] and a [`BaseDom`][`Domain`].
    ///
    //// Match a targetted[`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use (`onto`)[`Onto::onto`] of [`Self`].
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &Self::Item, target: &Mixed) -> Result<Self::TargetItem, OntoError> {
        match target {
            Mixed::Real(d) => match self.onto(item, d) {
                Ok(i) => Ok(MixedTypeDom::Real(i)),
                Err(e) => Err(e),
            },
            Mixed::Nat(d) => match self.onto(item, d) {
                Ok(i) => Ok(MixedTypeDom::Nat(i)),
                Err(e) => Err(e),
            },
            Mixed::Int(d) => match self.onto(item, d) {
                Ok(i) => Ok(MixedTypeDom::Int(i)),
                Err(e) => Err(e),
            },
            Mixed::Unit(d) => match self.onto(item, d) {
                Ok(i) => Ok(MixedTypeDom::Unit(i)),
                Err(e) => Err(e),
            },
            Mixed::Bool(_d) => unreachable!(
                "Converting a value from Unit onto Unit is not implemented, and it should not occur."
            ),
            Mixed::Cat(_d) => unreachable!(
                "Converting a value from Cat onto Cat is not implemented, and it should not occur."
            ),
        }
    }
}
impl OntoDom<Mixed> for Cat {}

impl From<Mixed> for Cat {
    fn from(value: Mixed) -> Self {
        match value {
            Mixed::Cat(d) => d,
            _ => unreachable!("Can only From<BaseDom> with Cat."),
        }
    }
}

impl Onto<Cat> for Cat {
    type Item = TypeDom<Cat>;
    type TargetItem = TypeDom<Cat>;
    /// [`Onto`] function between a [`Cat`] and a [`Cat`] [`Domain`].
    ///
    /// Considering $i$ the index of the item within `values` of [`Cat`]
    /// and $\ell_{en}$ the length of `values` of the [`Cat`] [`Domain`].
    /// The mapping is given by :
    ///
    /// $$ \frac{i+1}{\ell_{en}} $$
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &Self::Item, target: &Cat) -> Result<Self::TargetItem, OntoError> {
        let idx = self.values.iter().position(|n| n == item);

        match idx {
            Some(i) => {
                let mapped = target.values[i].clone();

                if target.is_in(&mapped) {
                    Ok(mapped)
                } else {
                    Err(OntoError(format!("{} input not in {}", item, self)))
                }
            }
            None => Err(OntoError(format!("{} input not in {}", item, self))),
        }
    }
}
impl OntoDom<Cat> for Cat {}

impl CSVWritable<(), <Cat as Domain>::TypeDom> for Cat {
    fn header(_elem: &()) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, comp: &<Cat as Domain>::TypeDom) -> Vec<String> {
        Vec::from([comp.to_string()])
    }
}
