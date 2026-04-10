use crate::{
    Bool, Bounded, CSVWritable, Domain, GridDomDistribution, Mixed, MixedTypeDom, Onto, OntoDom,
    Sampler, Unit,
    domain::{PreDomain, TypeDom, bounded::BoundedBounds},
    errors::OntoError,
};
use num::cast::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

/// A shortcut for the bounds of the generic type `<T>` in [`GridDom`]`<T>`
pub trait GridBounds:
    PartialEq + Clone + Display + Debug + Default + Serialize + for<'a> Deserialize<'a>
{
}
impl<T> GridBounds for T where
    T: PartialEq + Clone + Display + Debug + Default + Serialize + for<'a> Deserialize<'a>
{
}

/// A [`GridDom`] represents a discrete set of values for a given type `T` that can be sampled from.
/// It is a specific implementation of a [`Domain`] that allows for sampling from a predefined list of values
/// rather than a continuous range. This is particularly useful for optimization problems where the search space
/// is inherently discrete or when a user wants to restrict the search to specific values.
/// For example Grid Search is a common technique in hyperparameter optimization where a finite set of hyperparameter values are evaluated.
///
/// # Examples
/// ```
/// use tantale::core::{Domain, GridDom, Uniform};
/// let grid = GridDom::<f64>::new([0.1, 0.5, 0.9], Uniform);
/// let mut rng = rand::rng();
/// let sample = grid.sample(&mut rng);
/// assert!(grid.is_in(&sample));
/// let values = grid.values.to_vec();
/// assert_eq!(values, vec![0.1, 0.5, 0.9])
/// ```
#[derive(Debug, Clone)]
pub struct GridDom<T: GridBounds> {
    pub values: Box<[T]>,
    pub sampler: GridDomDistribution,
}

impl<T: GridBounds> GridDom<T> {
    /// Fabric for a [`GridDom`] [`Domain`].
    ///
    /// # Parameters
    /// * `values` - An iterable of values that can be converted into `T` that defines the discrete set of values for the [`GridDom`] domain.
    /// * `sampler` - A sampling strategy that defines how values are sampled from the `values` list. It is converted into
    ///   a [`GridDomDistribution`] that is used internally by the `sample` method of the [`GridDom`] domain.
    ///
    /// # Examples
    /// ```
    /// use tantale::core::{Domain, GridDom, Uniform};
    /// let grid = GridDom::<f64>::new([0.1, 0.5, 0.9], Uniform);
    ///
    /// let mut rng = rand::rng();
    /// let sample = grid.sample(&mut rng);
    /// assert!(grid.is_in(&sample));
    /// let values = grid.values.to_vec();
    /// assert_eq!(values, vec![0.1, 0.5, 0.9])
    /// ```
    pub fn new<
        I: IntoIterator<Item = Item>,
        Item: Into<T>,
        S: Sampler<Self> + Into<GridDomDistribution>,
    >(
        values: I,
        sampler: S,
    ) -> Self {
        Self {
            values: values.into_iter().map(Into::into).collect(),
            sampler: sampler.into(),
        }
    }

    /// Similar to [`GridDom::new`]. Used by the `hpo!` macro to create a [`GridDom`] from a list of values and a sampler.
    pub fn grid<
        I: IntoIterator<Item = Item>,
        Item: Into<T>,
        S: Sampler<GridDom<T>> + Into<GridDomDistribution>,
    >(
        values: I,
        sampler: S,
    ) -> GridDom<T> {
        GridDom::new(values, sampler)
    }
}

impl<T: GridBounds> PartialEq for GridDom<T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl<T: GridBounds> PreDomain for GridDom<T> {}
impl<T: GridBounds> Domain for GridDom<T> {
    type TypeDom = T;

    fn sample<R: rand::Rng>(&self, rng: &mut R) -> Self::TypeDom {
        let idx = rng.random_range(0..self.values.len());
        self.values[idx].clone()
    }

    fn is_in(&self, point: &Self::TypeDom) -> bool {
        self.values.contains(point)
    }
}

impl<T: GridBounds> Display for GridDom<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Grid({:?})", self.values)
    }
}

impl<Out, In> Onto<Bounded<Out>> for GridDom<In>
where
    In: GridBounds,
    Out: BoundedBounds,
    f64: AsPrimitive<Out>,
{
    type Item = In;
    type TargetItem = Out;
    /// [`Onto`] function between a [`GridDom`] and a [`Bounded`] [`Domain`].
    ///
    /// Considering $i$ the index of the item within `values` of [`GridDom`]
    /// and $\ell_{en}$ the length of `values` of the [`GridDom`] [`Domain`].
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
impl<In, Out> OntoDom<Bounded<Out>> for GridDom<In>
where
    In: GridBounds,
    Out: BoundedBounds,
    f64: AsPrimitive<Out>,
{
}

impl<In: GridBounds> Onto<Unit> for GridDom<In> {
    type Item = In;
    type TargetItem = TypeDom<Unit>;
    /// [`Onto`] function between a [`GridDom`] and a [`Unit`] [`Domain`].
    ///
    /// Considering $i$ the index of the item within `values` of [`GridDom`]
    /// and $\ell_{en}$ the length of `values` of the [`GridDom`] [`Domain`].
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
impl<In: GridBounds> OntoDom<Unit> for GridDom<In> {}

impl<In: GridBounds> Onto<Mixed> for GridDom<In> {
    type Item = In;
    type TargetItem = TypeDom<Mixed>;
    /// [`Onto`] function between a [`GridDom`] and a [`Mixed`] [`Domain`].
    ///
    /// Dispatches to the appropriate [`onto`](`Onto::onto`) method depending on the target [`Mixed`]
    /// sub-domain: [`Real`](`crate::domain::bounded::Real`), [`Int`](`crate::domain::bounded::Int`),
    /// [`Nat`](`crate::domain::bounded::Nat`), or [`Unit`].
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` - A borrowed targetted [`Mixed`] domain.
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///     * if the `target` [`Mixed`] variant is not [`Real`](`crate::domain::bounded::Real`), [`Int`](`crate::domain::bounded::Int`), [`Nat`](`crate::domain::bounded::Nat`), or [`Unit`].
    fn onto(&self, item: &Self::Item, target: &Mixed) -> Result<Self::TargetItem, OntoError> {
        match target {
            Mixed::Real(dom) => self.onto(item, dom).map(MixedTypeDom::Real),
            Mixed::Int(dom) => self.onto(item, dom).map(MixedTypeDom::Int),
            Mixed::Nat(dom) => self.onto(item, dom).map(MixedTypeDom::Nat),
            Mixed::Unit(dom) => self.onto(item, dom).map(MixedTypeDom::Unit),
            _ => Err(OntoError(format!(
                "Converting the value {:?} from {:?} onto Mixed is not implemented, and it should not occur.",
                item, self
            ))),
        }
    }
}

/// [`Onto`] function between a [`GridDom`]`<Out>` and another [`GridDom`]`<In>` [`Domain`].
///
/// Considering $i$ the index of the item within `values` of the source [`GridDom`] and $\ell_{en}$ the length of `values` of the source [`GridDom`] [`Domain`].
/// The index of the input `item` is mapped to the same index of the `target` [`GridDom`]`<Out>` [`Domain`] if the two [`GridDom`] have the same length, otherwise an error is returned.
impl<In: GridBounds, Out: GridBounds> Onto<GridDom<Out>> for GridDom<In> {
    type Item = In;
    type TargetItem = Out;
    /// [`Onto`] function between a source [`GridDom`]`<In>` and a target [`GridDom`]`<Out>` [`Domain`].
    ///
    /// The index of the input `item` within the source [`GridDom`] is mapped to the value at the
    /// same index in the `target` [`GridDom`]. Both grids must have the same length.
    ///
    /// # Parameters
    ///
    /// * `item` - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` - A borrowed targetted [`GridDom`]`<Out>`.
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if the source and `target` [`GridDom`] have different lengths.
    fn onto(
        &self,
        item: &Self::Item,
        target: &GridDom<Out>,
    ) -> Result<Self::TargetItem, OntoError> {
        if self.values.len() != target.values.len() {
            return Err(OntoError(format!(
                "Cannot map between GridDoms of different lengths: {} and {}",
                self.values.len(),
                target.values.len()
            )));
        }

        let idx = self.values.iter().position(|n| n == item);
        match idx {
            Some(i) => Ok(target.values[i].clone()),
            None => Err(OntoError(format!("{} input not in {}", item, self))),
        }
    }
}
impl<In: GridBounds, Out: GridBounds> OntoDom<GridDom<Out>> for GridDom<In> {}

impl<T: GridBounds> CSVWritable<(), T> for GridDom<T> {
    fn header(_elem: &()) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, comp: &T) -> Vec<String> {
        Vec::from([comp.to_string()])
    }
}

/// [`GridDom`] alias for a list of `f64` elements.
///
/// # Attributes
///
/// * `values` - A list of `f64` values defining to sample from.
///
/// # Examples
///
/// ```
/// use tantale::core::{GridReal,Domain,Uniform};
/// let dom = GridReal::new([1.0, 2.0, 3.0],Uniform);
///
/// let mut rng = rand::rng();
/// let sample = dom.sample(&mut rng);
/// assert!(dom.is_in(&sample));
/// assert_eq!(dom.values.to_vec(), vec![1.0, 2.0, 3.0]);
/// ```
pub type GridReal = GridDom<f64>;

impl From<Mixed> for GridReal {
    fn from(value: Mixed) -> Self {
        match value {
            Mixed::GridReal(d) => d,
            _ => unreachable!(
                "From<Mixed> for GridReal should only be called with Mixed::GridReal variant"
            ),
        }
    }
}

/// [`GridDom`] alias for a list of `i64` elements.
///
/// # Attributes
///
/// * `values` - A list of `i64` values defining to sample from.
///
/// # Examples
///
/// ```
/// use tantale::core::{GridInt,Domain,Uniform};
/// let dom = GridInt::new([-1, 0, 1],Uniform);
///
/// let mut rng = rand::rng();
/// let sample = dom.sample(&mut rng);
/// assert!(dom.is_in(&sample));
/// assert_eq!(dom.values.to_vec(), vec![-1, 0, 1]);
/// ```
pub type GridInt = GridDom<i64>;

impl From<Mixed> for GridInt {
    fn from(value: Mixed) -> Self {
        match value {
            Mixed::GridInt(d) => d,
            _ => unreachable!(
                "From<Mixed> for GridInt should only be called with Mixed::GridInt variant"
            ),
        }
    }
}

/// [`GridDom`] alias for a list of `u64` elements.
///
/// # Attributes
///
/// * `values` - A list of `u64` values defining to sample from.
///
/// # Examples
///
/// ```
/// use tantale::core::{GridNat,Domain,Uniform};
/// let dom = GridNat::new([1u64, 2, 3],Uniform);
///
/// let mut rng = rand::rng();
/// let sample = dom.sample(&mut rng);
/// assert!(dom.is_in(&sample));
/// assert_eq!(dom.values.to_vec(), vec![1, 2, 3]);
/// ```
pub type GridNat = GridDom<u64>;

impl From<Mixed> for GridNat {
    fn from(value: Mixed) -> Self {
        match value {
            Mixed::GridNat(d) => d,
            _ => unreachable!(
                "From<Mixed> for GridNat should only be called with Mixed::GridNat variant"
            ),
        }
    }
}

/// [`GridDom`] alias for a list of `String` elements.
///
/// # Attributes
///
/// * `values` - A list of `String` values defining to sample from.
///
/// # Examples
///
/// ```
/// use tantale::core::{Cat,Domain,Uniform};
/// let dom = Cat::new(["relu", "sigmoid", "tanh"],Uniform);
///
/// let mut rng = rand::rng();
/// let sample = dom.sample(&mut rng);
/// assert!(dom.is_in(&sample));
/// assert_eq!(dom.values.to_vec(), vec!["relu", "sigmoid", "tanh"]);
/// ```
pub type Cat = GridDom<String>;

impl From<Mixed> for Cat {
    fn from(value: Mixed) -> Self {
        match value {
            Mixed::Cat(d) => d,
            _ => unreachable!("From<Mixed> for Cat should only be called with Mixed::Cat variant"),
        }
    }
}

/// A type-erased discrete [`Domain`] grouping [`GridReal`], [`GridInt`], [`GridNat`], [`Cat`], and [`Bool`]
/// into a single enum. Unlike [`Mixed`], which covers all basic domains, [`Grid`] is restricted to
/// grid-based domains and [`Bool`], making it well-suited for exhaustive enumeration strategies
/// such as Grid Search.
/// The [`TypeDom`](`Domain::TypeDom`) is a [`MixedTypeDom`].
#[derive(Clone, PartialEq)]
pub enum Grid {
    Real(GridReal),
    Int(GridInt),
    Nat(GridNat),
    Cat(Cat),
    Bool(Bool),
}

impl Grid {
    /// Get the value at the given index in the values, if it exists.
    pub fn get(&self, idx: usize) -> Option<MixedTypeDom> {
        match self {
            Grid::Real(dom) => dom.values.get(idx).map(|v| MixedTypeDom::Real(*v)),
            Grid::Int(dom) => dom.values.get(idx).map(|v| MixedTypeDom::Int(*v)),
            Grid::Nat(dom) => dom.values.get(idx).map(|v| MixedTypeDom::Nat(*v)),
            Grid::Cat(dom) => dom.values.get(idx).map(|v| MixedTypeDom::Cat(v.clone())),
            Grid::Bool(_) => {
                if idx == 0 {
                    Some(MixedTypeDom::Bool(false))
                } else if idx == 1 {
                    Some(MixedTypeDom::Bool(true))
                } else {
                    None
                }
            }
        }
    }

    /// Get the size of the grid, i.e. the number of values it contains.
    pub fn size(&self) -> usize {
        match self {
            Grid::Real(dom) => dom.values.len(),
            Grid::Int(dom) => dom.values.len(),
            Grid::Nat(dom) => dom.values.len(),
            Grid::Cat(dom) => dom.values.len(),
            Grid::Bool(_) => 2,
        }
    }
}

impl Display for Grid {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Real(d) => std::fmt::Display::fmt(&d, f),
            Self::Nat(d) => std::fmt::Display::fmt(&d, f),
            Self::Int(d) => std::fmt::Display::fmt(&d, f),
            Self::Bool(d) => std::fmt::Display::fmt(&d, f),
            Self::Cat(d) => std::fmt::Display::fmt(&d, f),
        }
    }
}
impl std::fmt::Debug for Grid {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Real(d) => std::fmt::Debug::fmt(&d, f),
            Self::Nat(d) => std::fmt::Debug::fmt(&d, f),
            Self::Int(d) => std::fmt::Debug::fmt(&d, f),
            Self::Bool(d) => std::fmt::Debug::fmt(&d, f),
            Self::Cat(d) => std::fmt::Debug::fmt(&d, f),
        }
    }
}

impl PreDomain for Grid {}
impl Domain for Grid {
    type TypeDom = MixedTypeDom;

    fn sample<R: rand::Rng>(&self, rng: &mut R) -> Self::TypeDom {
        match self {
            Grid::Real(dom) => MixedTypeDom::Real(dom.sample(rng)),
            Grid::Int(dom) => MixedTypeDom::Int(dom.sample(rng)),
            Grid::Nat(dom) => MixedTypeDom::Nat(dom.sample(rng)),
            Grid::Cat(dom) => MixedTypeDom::Cat(dom.sample(rng)),
            Grid::Bool(dom) => MixedTypeDom::Bool(dom.sample(rng)),
        }
    }

    fn is_in(&self, point: &Self::TypeDom) -> bool {
        match (self, point) {
            (Grid::Real(dom), MixedTypeDom::Real(p)) => dom.is_in(p),
            (Grid::Int(dom), MixedTypeDom::Int(p)) => dom.is_in(p),
            (Grid::Nat(dom), MixedTypeDom::Nat(p)) => dom.is_in(p),
            (Grid::Cat(dom), MixedTypeDom::Cat(p)) => dom.is_in(p),
            (Grid::Bool(dom), MixedTypeDom::Bool(p)) => dom.is_in(p),
            _ => false, // Type mismatch
        }
    }
}

impl CSVWritable<(), MixedTypeDom> for Grid {
    fn header(_elem: &()) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, comp: &MixedTypeDom) -> Vec<String> {
        match comp {
            MixedTypeDom::Real(s) => Vec::from([s.to_string()]),
            MixedTypeDom::Nat(s) => Vec::from([s.to_string()]),
            MixedTypeDom::Int(s) => Vec::from([s.to_string()]),
            MixedTypeDom::Bool(s) => Vec::from([s.to_string()]),
            MixedTypeDom::Cat(s) => Vec::from([s.to_string()]),
            MixedTypeDom::Unit(s) => Vec::from([s.to_string()]),
            MixedTypeDom::GridReal(s) => Vec::from([s.to_string()]),
            MixedTypeDom::GridNat(s) => Vec::from([s.to_string()]),
            MixedTypeDom::GridInt(s) => Vec::from([s.to_string()]),
        }
    }
}
