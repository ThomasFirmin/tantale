//! A [`Bounded`] domain defines a binary where values are in ${\texttt{false}, \texttt{true}}$.
//! For example, the `amsgrad` parameter of the [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)
//! optimizer in [Pytorch](https://pytorch.org/).
//!
//! # Example
//!
//! ```
//! use tantale::core::{Bool, Domain};
//!
//! let mut rng = rand::rng();
//! let dom = Bool::new();
//!
//! let sample = dom.sample(&mut rng);
//! assert!(dom.is_in(&sample));
//! ```
use crate::{
    domain::{
        Domain, PreDomain, TypeDom, base::{BaseDom, BaseTypeDom}, bounded::{Bounded, BoundedBounds}, onto::{Onto, OntoDom}, unit::Unit
    },
    errors::OntoError,
    recorder::csv::CSVWritable, sampler::{BoolDistribution, Sampler},
};

use num::cast::AsPrimitive;
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};
use std::fmt;

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
/// let sample = dom.sample(&mut rng);
/// assert!(dom.is_in(&sample));
/// ```
#[derive(Clone, Copy)]
pub struct Bool(pub BoolDistribution);
impl Bool {
    /// Fabric for a [`Bool`].
    pub fn new<S:Sampler<Self> + Into<BoolDistribution>>(sampler:S) -> Bool {
        Bool(sampler.into())
    }
}

impl PartialEq for Bool {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl PreDomain for Bool{}
impl Domain for Bool {
    type TypeDom = bool;

    /// Sample a `bool` using the inner [`BoolDistribution`] of [`Bool`].
    fn sample(&self, rng: &mut ThreadRng) -> Self::TypeDom {
        self.0.sample(self, rng)
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
    /// use tantale::core::{Bool, Domain};
    ///
    /// let mut rng = rand::rng();
    /// let dom = Bool::new();
    ///
    /// let sample = dom.sample(&mut rng);
    /// assert!(dom.is_in(&sample));
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
    Out: BoundedBounds + Serialize + for<'a> Deserialize<'a>,
    f64: AsPrimitive<Out>,
{
    type Item = TypeDom<Bool>;
    type TargetItem = TypeDom<Bounded<Out>>;
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
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &Self::Item,
        target: &Bounded<Out>,
    ) -> Result<Self::TargetItem, OntoError> {
        if self.is_in(item) {
            let mapped = if *item {
                target.bounds.end()
            } else {
                target.bounds.start()
            };
            if target.is_in(mapped) {
                Ok(*mapped)
            } else {
                Err(OntoError(format!("{} input not in {}", item, self)))
            }
        } else {
            Err(OntoError(format!("{} input not in {}", item, self)))
        }
    }
}
impl<Out> OntoDom<Bounded<Out>> for Bool
where
    Out: BoundedBounds + Serialize + for<'a> Deserialize<'a>,
    f64: AsPrimitive<Out>,
{
}

impl Onto<Unit> for Bool {
    type Item = TypeDom<Bool>;
    type TargetItem = TypeDom<Unit>;
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
    /// * Returns a [`OntoError`])
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &Self::Item, target: &Unit) -> Result<Self::TargetItem, OntoError> {
        if self.is_in(item) {
            let mapped = if *item { 1.0 } else { 0.0 };
            if target.is_in(&mapped) {
                Ok(mapped)
            } else {
                Err(OntoError(format!("{} input not in {}", item, self)))
            }
        } else {
            Err(OntoError(format!("{} input not in {}", item, self)))
        }
    }
}
impl OntoDom<Unit> for Bool {}

impl Onto<BaseDom> for Bool {
    type Item = TypeDom<Bool>;
    type TargetItem = TypeDom<BaseDom>;
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
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &Self::Item, target: &BaseDom) -> Result<Self::TargetItem, OntoError> {
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
impl OntoDom<BaseDom> for Bool {}

impl Onto<Bool> for Bool {
    type Item = TypeDom<Bool>;
    type TargetItem = TypeDom<Bool>;
    /// [`Onto`] function between a [`Bool`] [`Domain`] and a [`Bool`][`Domain`].
    ///
    /// Considering $x$ the `item`, returns a copy of $x$.
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Bool`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &Self::Item, _target: &Bool) -> Result<Self::TargetItem, OntoError> {
        Ok(*item)
    }
}
impl OntoDom<Bool> for Bool {}

impl From<BaseDom> for Bool {
    fn from(value: BaseDom) -> Self {
        match value {
            BaseDom::Bool(d) => d,
            _ => unreachable!("Can only From<BaseDom> with Bool."),
        }
    }
}

impl CSVWritable<(), bool> for Bool {
    fn header(_elem: &()) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, comp: &bool) -> Vec<String> {
        Vec::from([comp.to_string()])
    }
}
