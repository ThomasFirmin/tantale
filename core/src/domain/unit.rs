//! A [`Unit`] domain defines a `f64` [`Domain`] within $[0.0, 1.0]$.
//! It is similar to [`Bounded`] but with a simplified definition.
//!
//! # Examples
//!
//! ```
//! use tantale::core::{Unit, Domain, DomainBounded};
//! let dom = Unit::new();
//!
//! let mut rng = rand::rng();
//! let sample = dom.sample(&mut rng);
//! assert!(dom.is_in(&sample));
//! assert_eq!(dom.lower(), 0.0);
//! assert_eq!(dom.upper(), 1.0);
//! assert_eq!(dom.mid(), 0.5);
//! assert_eq!(dom.width(), 1.0);
//! ```

use crate::{
    domain::{
        base::{BaseDom, BaseTypeDom},
        bool::Bool,
        bounded::{Bounded, BoundedBounds},
        cat::Cat,
        onto::{Onto, OntoDom},
        Domain, TypeDom,
    },
    errors::OntoError,
    recorder::csv::CSVWritable,
};

use num::cast::AsPrimitive;
use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};
use std::{fmt, ops::RangeInclusive};

/// A [`f64`] [`Unit`] domain within `[0,1]`.
/// A generic [`Unit`] [`Domain`] with a numerical `lower=0.0` and `upper=1.0` bounds.
///
pub struct Unit<Sampler> {
    bounds: RangeInclusive<f64>,
    mid:f64,
    width:f64,
    sampler:
}

impl Unit {
    /// Fabric for a [`Unit`] [`Domain`].
    pub fn new() -> Unit {
        Unit {
            bounds: RangeInclusive::new(0.0, 1.0),
            mid:0.5,
            width:1.0,
        }
    }
}

impl Default for Unit {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Unit {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Domain for Unit {
    type TypeDom = f64;

    /// Default sampler for [`Unit`].
    /// See [`uniform`].
    fn sample(&self, rng: &mut ThreadRng) -> Self::TypeDom {
        uniform(&self.bounds, rng)
    }

    fn is_in(&self, item: &Self::TypeDom) -> bool {
        self.bounds.contains(item)
    }
}


impl std::clone::Clone for Unit {
    fn clone(&self) -> Self {
        Unit::new()
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}

impl fmt::Debug for Unit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}

impl<Out> Onto<Bounded<Out>> for Unit
where
    Out: BoundedBounds + Serialize + for<'a> Deserialize<'a>,
    f64: AsPrimitive<Out>,
{
    type Item = TypeDom<Unit>;
    type TargetItem = TypeDom<Bounded<Out>>;
    /// [`Onto`] function between a [`Unit`] [`Domain`] and a [`Bounded`] [`Domain`].
    ///
    /// Considering $l_{out}$, $u_{out}$ the lower and upper bounds of
    /// the output [`Bounded`] [`Domain`], and $x$ the item to be mapped,
    /// the mapping is given by
    ///
    /// $$ x \times (u_{out}-l_{out}) + l_{out}$$
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
            let a: f64 = *item;
            let c: f64 = target.width.as_();
            let mapped: Out = (a * c).as_() + *target.bounds.start();

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
impl<Out> OntoDom<Bounded<Out>> for Unit
where
    Out: BoundedBounds + Serialize + for<'a> Deserialize<'a>,
    f64: AsPrimitive<Out>,
{
}

impl Onto<Bool> for Unit
where
    f64: AsPrimitive<f64>,
{
    type Item = TypeDom<Unit>;
    type TargetItem = TypeDom<Bool>;
    /// [`Onto`] function between a [`Unit`] [`Domain`] and a [`Bool`][`Domain`].
    ///
    /// Considering $x$ the `item`, returns `true` if $x>0.5$
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
        if self.is_in(item) {
            Ok(*item > 0.5)
        } else {
            Err(OntoError(format!("{} input not in {}", item, self)))
        }
    }
}
impl OntoDom<Bool> for Unit {}

impl Onto<Cat> for Unit {
    type Item = TypeDom<Unit>;
    type TargetItem = TypeDom<Cat>;
    /// [`Onto`] function between a [`Unit`] [`Domain`] and a [`Cat`][`Domain`].
    ///
    /// Considering $\ell_{en}$ the length of `values` of the [`Cat`] [`Domain`].
    /// The variable $x$ is the item to be mapped.
    /// The mapping is given by mapping item to an index of `values` in [`Cat`]:
    ///
    /// $$ i = \\left\\lfloor x \times \ell_{en} \\right\\rfloor $$
    /// $$ \\texttt{index} = \\begin{cases}
    ///    i & \\text{if } i < \ell_{en} \\\\
    ///    i -1 & \\text{if } i = \ell_{en}
    /// \\end{cases} $$
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Cat`]<'a, N> - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &Self::Item, target: &Cat) -> Result<Self::TargetItem, OntoError> {
        if self.is_in(item) {
            let a: f64 = item.as_();
            let c: f64 = target.values.len().as_();
            let idx = (a * c) as usize;
            let idx = if idx == target.values.len() {
                idx - 1
            } else {
                idx
            };
            let mapped = target.values[idx].clone();

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
impl OntoDom<Cat> for Unit {}

impl Onto<BaseDom> for Unit {
    type Item = TypeDom<Unit>;
    type TargetItem = TypeDom<BaseDom>;
    /// [`Onto`] function between a [`Bounded`] [`Domain`] and a [`BaseDom`][`Domain`].
    ///
    //// Match a targetted[`Bounded`], [`Bool`], [`Cat`] and [`Unit`] and (`onto`)[`Onto::onto`] of [`Self`].
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
            BaseDom::Unit(_d) => unreachable!("Converting a value from Unit onto Unit is not implemented, and it should not occur."),
            BaseDom::Bool(d) => {
                match self.onto(item, d) {
                    Ok(i) => Ok(BaseTypeDom::Bool(i)),
                    Err(e) => Err(e),
                }
            },
            BaseDom::Cat(d) => {
                match self.onto(item, d) {
                    Ok(i) => Ok(BaseTypeDom::Cat(i)),
                    Err(e) => Err(e),
                }
            },
        }
    }
}
impl OntoDom<BaseDom> for Unit {}

impl From<BaseDom> for Unit {
    fn from(value: BaseDom) -> Self {
        match value {
            BaseDom::Unit(d) => d,
            _ => unreachable!("Can only From<BaseDom> with Unit."),
        }
    }
}

impl Onto<Unit> for Unit
where
    f64: AsPrimitive<f64>,
{
    type Item = TypeDom<Unit>;
    type TargetItem = TypeDom<Unit>;
    /// [`Onto`] function between a [`Unit`] [`Domain`] and a [`Unit`][`Domain`].
    ///
    /// Considering $x$ the `item`, returns a copy of $x$.
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
    fn onto(&self, item: &Self::Item, _target: &Unit) -> Result<Self::TargetItem, OntoError> {
        Ok(*item)
    }
}
impl OntoDom<Unit> for Unit {}

impl CSVWritable<(), f64> for Unit {
    fn header(_elem: &()) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, comp: &f64) -> Vec<String> {
        Vec::from([comp.to_string()])
    }
}
