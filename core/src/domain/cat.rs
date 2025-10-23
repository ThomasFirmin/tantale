//! A [`Cat`] domain defines a domain that is made of categorical
//! and non ordinal features.
//! For example, let's take the [activation function](https://en.wikipedia.org/w/index.php?title=Activation_function&oldid=1287429111),
//! of a neural network, the features can be `["relu", "tanh", "sigmoid"]`.
//!
//! # Examples
//!
//! ```
//! use tantale::core::{Cat,Domain};
//!
//! let mut rng = rand::rng();
//! let check = ["relu", "tanh", "sigmoid"];
//! let dom = Cat::new(&["relu", "tanh", "sigmoid"]);
//!
//! let sample = dom.sample(&mut rng);
//! assert!(dom.is_in(&sample));
//! assert_eq!(dom.values(), &check);
//! ```
use crate::domain::{
    Domain, TypeDom, base::{BaseDom, BaseTypeDom}, bounded::{Bounded, BoundedBounds, DomainBounded}, derrors::OntoError, onto::{Onto, OntoDom}, sampler::uniform_cat, unit::Unit
};
use crate::saver::CSVWritable;

use num::cast::AsPrimitive;
use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};
use std::fmt;
// _-_-_-_-_-_-__-_-_-_-_-_-_-_
// Categorical domain

/// Describes a non-ordinal categorical domain. It is made of features,
/// described by the private attribute `values`, an [`array`] of [`str`].
/// Each elements describes a unique feature.
/// The values can be accessed by the corresponding getter `values()`.
///
/// # Attributes
///
///  * `values` : `[Vec]<[String]>` - A static array of the features defining the categorical [`Domain`].
/// ```
#[derive(Clone)]
pub struct Cat {
    values: Vec<String>,
}
impl Cat {
    /// Fabric for a [`Cat`].
    ///
    /// # Attributes
    ///
    ///  * `values` : `&'a [&'a str]` - A static array of the features defining the categorical [`Domain`].
    ///
    pub fn new<'a>(values: &'a [&'a str]) -> Cat {
        Cat {
            values: values.iter().map(|s| String::from(*s)).collect(),
        }
    }
    /// Getter for values
    pub fn values(&self) -> &Vec<String> {
        &self.values
    }
}

impl PartialEq for Cat {
    fn eq(&self, other: &Self) -> bool {
        self.values() == other.values()
    }
}

impl Domain for Cat {
    type TypeDom = String;

    /// Default sampler for [`Cat`] is a uniform choice within the `values`
    /// See [`uniform_cat`].
    fn sample(&self, rng: &mut ThreadRng) -> TypeDom<Self> {
        uniform_cat(self, rng)
    }

    /// Method to check if a given point is in the domain.
    ///
    /// # Attributes
    ///
    /// * `point` : `&`[`Self`]`::`[`TypeDom`](Domain::TypeDom) :
    ///   a point of the same type as the type of the domain.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Cat,Domain};
    ///
    /// let mut rng = rand::rng();
    /// static ACTIVATION : [&str; 3] = ["relu", "tanh", "sigmoid"];
    ///
    /// let mut rng = rand::rng();
    /// let check = ["relu", "tanh", "sigmoid"];
    /// let dom = Cat::new(&ACTIVATION);
    ///
    /// let sample = dom.sample(&mut rng);
    /// assert!(dom.is_in(&sample));
    /// assert_eq!(dom.values(), &check);
    ///
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
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Bounded`]`<Out>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &Self::Item, target: &Bounded<Out>) -> Result<Self::TargetItem, OntoError>{
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
{}

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
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Unit`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &Self::Item, target: &Unit) -> Result<Self::TargetItem, OntoError>{
        let idx = self.values().iter().position(|n| n == item);

        match idx {
            Some(i) => {
                let a: f64 = (i + 1).as_();
                let b: f64 = self.values().len().as_();
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
impl OntoDom<Unit> for Cat{}

impl Onto<BaseDom> for Cat {
    type Item = TypeDom<Cat>;
    type TargetItem = TypeDom<BaseDom>;
    /// [`Onto`] function between a [`Cat`] [`Domain`] and a [`BaseDom`][`Domain`].
    ///
    //// Match a targetted[`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and use (`onto`)[`Onto::onto`] of [`Self`].
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`BaseDom`]`<N,T>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &Self::Item, target: &BaseDom) -> Result<Self::TargetItem, OntoError>{
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
impl OntoDom<BaseDom> for Cat{}

impl From<BaseDom> for Cat {
    fn from(value: BaseDom) -> Self {
        match value {
            BaseDom::Cat(d) => d,
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
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Cat`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &Self::Item, target: &Cat) -> Result<Self::TargetItem, OntoError>{
        let idx = self.values().iter().position(|n| n == item);

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
impl OntoDom<Cat> for Cat{}

impl CSVWritable<(), <Cat as Domain>::TypeDom> for Cat {
    fn header(_elem: &()) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, comp: &<Cat as Domain>::TypeDom) -> Vec<String> {
        Vec::from([comp.to_string()])
    }
}
