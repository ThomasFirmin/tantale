use crate::{
    Grid, distribution::{BoolDistribution, Distribution}, domain::{
        CategoricalDomain, Domain, Elem, Element, PreDomain, bounded::{Bounded, BoundedBounds}, mixed::{Mixed, MixedTypeDom}, onto::{Onto, OntoDom}, unit::Unit,
    }, errors::OntoError,
};

use num::cast::AsPrimitive;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A [`Bool`] domain defines a [`Domain`] for which values are in ${\texttt{false}, \texttt{true}}$.
///
/// # Examples
///
/// ```
/// use tantale::core::{Bool,Domain, Bernoulli};
/// let dom = Bool::new(Bernoulli(0.5));
///
/// let mut rng = rand::rng();
/// let sample = dom.sample(&mut rng);
/// assert!(dom.contains(&sample));
/// ```
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Bool(pub BoolDistribution);
impl Bool {
    /// Fabric for a [`Bool`].
    pub fn new<S: Distribution<Self> + Into<BoolDistribution>>(sampler: S) -> Bool {
        Bool(sampler.into())
    }
    // Fabric for a [`Bool`] dedicated to be wrapped in a [`Grid`](crate::domain::grid::Grid).
    pub fn grid<S: Distribution<Self> + Into<BoolDistribution>>(sampler: S) -> Bool {
        Bool(sampler.into())
    }
}

impl PartialEq for Bool {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl PreDomain for Bool {}
impl Domain for Bool {
    type Element = bool;

    /// Sample a `bool` using the inner [`BoolDistribution`] of [`Bool`].
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Element {
        self.0.sample(self, rng)
    }

    /// Method to check if a given point is in the domain.
    ///
    /// # Parameters
    ///
    /// * `point` : `&`[`TypeDom`](Domain::TypeDom) - a borrowed sample from a [`Bool`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Bool,Domain, Bernoulli};
    /// let dom = Bool::new(Bernoulli(0.5));
    ///
    /// let mut rng = rand::rng();
    /// let sample = dom.sample(&mut rng);
    /// assert!(dom.contains(&sample));
    /// ```
    fn contains(&self, _point: &Self::Element) -> bool {
        true
    }
}

impl CategoricalDomain for Bool {
    fn size(&self) -> usize {
        2
    }

    fn get_features(&self) -> &[Self::Element] {
        &[false, true]
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
    type Item = Elem<Bool>;
    type TargetItem = Elem<Bounded<Out>>;
    /// [`Onto`] function between a [`Bool`] and a [`Bounded`] [`Domain`].
    ///
    /// # Parameters
    ///
    /// * `item` : [`bool`] - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Bool`].
    /// * `target` : `&`[`Bounded`] - A borrowed targetted [`Bounded`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Bool`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Bounded`] domain.
    fn onto(
        &self,
        item: &Self::Item,
        target: &Bounded<Out>,
    ) -> Result<Self::TargetItem, OntoError> {
        if self.contains(item) {
            let mapped = if *item {
                target.bounds.end()
            } else {
                target.bounds.start()
            };
            if target.contains(mapped) {
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
    type Item = Elem<Bool>;
    type TargetItem = Elem<Unit>;
    /// [`Onto`] function between a [`Bool`] and a [`Unit`] [`Domain`].
    ///
    /// # Parameters
    ///
    /// * `item` : [`bool`] - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Bool`].
    /// * `target` : `&`[`Unit`] - A borrowed targetted [`Unit`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Bool`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Unit`] domain.
    fn onto(&self, item: &Self::Item, target: &Unit) -> Result<Self::TargetItem, OntoError> {
        if self.contains(item) {
            let mapped = if *item { 1.0 } else { 0.0 };
            if target.contains(&mapped) {
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

impl Onto<Mixed> for Bool {
    type Item = Elem<Bool>;
    type TargetItem = Elem<Mixed>;
    /// [`Onto`] function between a [`Bool`] and a [`Mixed`] [`Domain`].
    ///
    /// Uses the respective [`onto`](Onto::onto) functions for each possible target [`Domain`] inside the [`Mixed`].
    ///
    /// # Parameters
    ///
    /// * `item` : [`bool`] - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Bool`].
    /// * `target` : `&`[`Mixed`] - A borrowed targetted [`Mixed`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if [`Onto::Item`] to be mapped is not into [`Bool`] domain.
    ///     * if [`Onto::TargetItem`] is not into the [`Mixed`] domain.
    fn onto(&self, item: &Self::Item, target: &Mixed) -> Result<Self::TargetItem, OntoError> {
        match target {
            Mixed::Real(d) => self.onto(item, d).map(MixedTypeDom::Real),
            Mixed::Nat(d) => self.onto(item, d).map(MixedTypeDom::Nat),
            Mixed::Int(d) => self.onto(item, d).map(MixedTypeDom::Int),
            Mixed::Unit(d) => self.onto(item, d).map(MixedTypeDom::Unit),
            _ => Err(OntoError(format!("Cannot map Bool to {:?} domain", target))),
        }
    }
}
impl OntoDom<Mixed> for Bool {}

impl Onto<Bool> for Bool {
    type Item = Elem<Bool>;
    type TargetItem = Elem<Bool>;
    /// [`Onto`] function between a [`Bool`] and another [`Bool`] [`Domain`].
    ///
    /// # Parameters
    ///
    /// * `item` : [`bool`] - A borrowed [`TypeDom`](Domain::TypeDom) from the [`Bool`].
    /// * `target` : `&`[`Bool`] - A borrowed targetted [`Bool`].
    fn onto(&self, item: &Self::Item, _target: &Bool) -> Result<Self::TargetItem, OntoError> {
        Ok(*item)
    }
}
impl OntoDom<Bool> for Bool {}

impl From<Mixed> for Bool {
    fn from(value: Mixed) -> Self {
        match value {
            Mixed::Bool(d) => d,
            _ => unreachable!("Can only From<BaseDom> with Bool."),
        }
    }
}

impl From<Grid> for Bool {
    fn from(value: Grid) -> Self {
        match value {
            Grid::Bool(d) => d,
            _ => unreachable!("Can only From<Grid> with Bool."),
        }
    }
}