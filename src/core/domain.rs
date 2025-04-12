#[doc(alias = "Domain")]
/// # Domain
/// This crate describes what a domain of a variable is.
/// Most of the domains implements the [`Domain`] type trait [`TypeDom`].
/// It gives the type of a point within this domain.
/// Domains are use in [`crate::core::variable::Variable`] to define the type of the variable,
/// its `TypeObjective` and `TypeOptimizer`, repectively the input type of that variable within
/// the [`crate::core::objective:Objective`] function, and the input type of the
/// [`crate::core::optimizer::Optimizer`].
///
use crate::core::sampler::{uniform, uniform_bool, uniform_cat};

use num::{Float, Num, NumCast};
use rand::distr::uniform::SampleUniform;
use rand::prelude::ThreadRng;
use std::fmt::{self, Debug, Display};
use std::ops::RangeInclusive;

/// [`Domain`] is a trait describing the type of a point from the domain it is attached to.
/// It must implement the `default_sampler` and `is_in` methods.
pub trait Domain {
    type TypeDom:PartialEq + Clone + Copy + Display + Debug;
    /// Associated function to automatically return a default [`crate::core::sampler`]
    /// for the domain the trait is implemented.
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom;
    /// Returns `true` if a given borrowed `point` is in the domain. Otherwise returns `false`.
    ///
    /// # Parameters
    ///
    /// * `point` : `&`[`Self`]`::`[`TypeDom`](Domain::TypeDom) - a borrowed point from the [`Domain`].
    ///
    fn is_in(&self, point: &Self::TypeDom) -> bool;
}

/// Describes a peculiar trait for some of the domains that are numerically bounded by a
/// `lower` and `upper` bound.
pub trait DomainBounded: Domain {
    /// Getter method for the lower bound.
    fn lower(&self) -> Self::TypeDom;
    /// Getter method for the upper bound.
    fn upper(&self) -> Self::TypeDom;
    /// Getter method for the middle point of the domain.
    fn mid(&self) -> Self::TypeDom;
    /// Getter method for the width of the domain.
    fn width(&self) -> Self::TypeDom;
}

/// A generic [`Bounded`] [`Domain`] with a numerical `lower` and `upper` bounds.
///
/// # Arguments
/// * `bounds`: [`RangeInclusive`]`<T>` - A [`RangeInclusive`] object of type `<T>`.
/// * `mid`: `T` - Middle point of the [`Bounded`] [`Domain`]. $\frac{\texttt{lower}+\texttt{upper}}{2}$
/// * `width`: `T` - Width of the [`Bounded`] [`Domain`]. $\texttt{upper}-\texttt{lower}$
///
pub struct Bounded<T>
where
    T: Num + NumCast,
    T: PartialEq + PartialOrd,
    T: Clone + Copy,
    T: Display+Debug,
{
    bounds: RangeInclusive<T>,
    mid: T,
    width: T,
}

impl<T> Bounded<T>
where
    T: Num + NumCast,
    T: PartialEq + PartialOrd,
    T: SampleUniform,
    T: Clone + Copy,
    T: Display+Debug,
{
    /// Fabric for a [`Bounded`].
    ///
    /// * The `mid` attribute is automatically computed with $\frac{\texttt{lower}+\texttt{upper}}{2}$.
    /// * The `width` attribute is automatically computed with $\texttt{upper}-\texttt{lower}$.
    ///
    /// # Parameters
    /// * `lower`: `T` - Lower bound of the [`Bounded`] [`Domain`].
    /// * `upper`: `T` - Upper bound of the [`Bounded`] [`Domain`].
    ///
    pub fn new(lower: T, upper: T) -> Bounded<T> {
        if lower < upper {
            let mid = (upper.clone() + lower.clone()) / T::from(2).unwrap();
            let width = upper.clone() - lower.clone();
            Bounded {
                bounds: std::ops::RangeInclusive::new(lower, upper),
                mid,
                width,
            }
        } else {
            panic!("Boundaries error, {} is not < {}.", lower, upper);
        }
    }
}

impl<T> Domain for Bounded<T>
where
    T: Num + NumCast,
    T: PartialEq + PartialOrd,
    T: SampleUniform,
    T: Clone + Copy,
    T: Display+Debug,
{
    type TypeDom = T;

    /// Default sampler for [`Bounded`].
    /// See [`uniform`].
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        |s, rng| uniform(&s.bounds, rng)
    }

    fn is_in(&self, item: &T) -> bool {
        self.bounds.contains(item)
    }
}

impl<T> DomainBounded for Bounded<T>
where
    T: Num + NumCast,
    T: PartialEq + PartialOrd,
    T: SampleUniform,
    T: Clone + Copy,
    T: Display+Debug,
{
    fn lower(&self) -> Self::TypeDom {
        self.bounds.start().clone()
    }
    fn upper(&self) -> Self::TypeDom {
        self.bounds.end().clone()
    }
    fn mid(&self) -> Self::TypeDom {
        self.mid.clone()
    }
    fn width(&self) -> Self::TypeDom {
        self.width.clone()
    }
}

impl<T> std::clone::Clone for Bounded<T>
where
    T: Num + NumCast,
    T: PartialEq + PartialOrd,
    T: SampleUniform,
    T: Clone + Copy,
    T: Display+Debug,
{
    fn clone(&self) -> Self {
        Bounded::new(self.bounds.start().clone(), self.bounds.end().clone())
    }
}

impl<T> fmt::Display for Bounded<T>
where
    T: Num + NumCast,
    T: PartialEq + PartialOrd,
    T: SampleUniform,
    T: Clone + Copy,
    T: Display+Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}

/// [`Bounded`] alias for a continuous `f64` [`Domain`] bounded by a lower and upper bounds.
///
/// # Attributes
///
/// For safety, the attributes are private and can only be accessed via getter methods
/// `.lower()`, `.upper()`, `.mid()` and `.width()`.
/// It prevents any modification of the [`Domain`] during the optimization process.
/// Some algorithms might modify the [`Domain`], for example *divide-and-conquer* approaches.
/// In that case, the algorithm should create a new [`Domain`].
///
/// * `lower`: `f64` - Lower bound of the [`Real`] [`Domain`].
/// * `upper`: `f64` - Upper bound of the [`Real`] [`Domain`].
/// * `mid`: `f64` - Middle point of the [`Real`] [`Domain`].
/// * `width`: `f64` - Width of the [`Real`] [`Domain`].
///
/// # Examples
///
/// ```
/// use tantale::core::{Real,Domain,DomainBounded};
/// let dom = Real::new(0.0, 10.0);
///
/// let mut rng = rand::rng();
/// let sampler = dom.default_sampler();
/// assert!(dom.is_in(&sampler(&dom, &mut rng)));
/// assert_eq!(dom.lower(), 0.0);
/// assert_eq!(dom.upper(), 10.0);
/// assert_eq!(dom.mid(), 5.0);
/// assert_eq!(dom.width(), 10.0);
/// ```
pub type Real = Bounded<f64>;

/// [`Bounded`] alias for a natural `u64` [`Domain`] bounded by a lower and upper bounds.
///
/// # Attributes
///
/// For safety, the attributes are private and can only be accessed via getter methods
/// `.lower()`, `.upper()`, `.mid()` and `.width()`.
/// It prevents any modification of the [`Domain`] during the optimization process.
/// Some algorithms might modify the [`Domain`], for example *divide-and-conquer* approaches.
/// In that case, the algorithm should create a new [`Domain`].
///
/// * `lower`: `u64` - Lower bound of the [`Nat`] [`Domain`].
/// * `upper`: `u64` - Upper bound of the [`Nat`] [`Domain`].
/// * `mid`: `u64` - Middle point of the [`Nat`] [`Domain`].
/// * `width`: `u64` - Width of the [`Nat`] [`Domain`].
///
/// # Examples
///
/// ```
/// use tantale::core::{Nat,Domain,DomainBounded};
/// let dom = Nat::new(0, 10);
///
/// let mut rng = rand::rng();
/// let sampler = dom.default_sampler();
/// assert!(dom.is_in(&sampler(&dom, &mut rng)));
/// assert_eq!(dom.lower(), 0);
/// assert_eq!(dom.upper(), 10);
/// assert_eq!(dom.mid(), 5);
/// assert_eq!(dom.width(), 10);
/// ```
pub type Nat = Bounded<u64>;

/// [`Bounded`] alias for an integer `i64` [`Domain`] bounded by a lower and upper bounds.
///
/// # Attributes
///
/// For safety, the attributes are private and can only be accessed via getter methods
/// `.lower()`, `.upper()`, `.mid()` and `.width()`.
/// It prevents any modification of the [`Domain`] during the optimization process.
/// Some algorithms might modify the [`Domain`], for example *divide-and-conquer* approaches.
/// In that case, the algorithm should create a new [`Domain`].
///
/// * `lower`: `i64` - Lower bound of the [`Int`] [`Domain`].
/// * `upper`: `i64` - Upper bound of the [`Int`] [`Domain`].
/// * `mid`: `i64` - Middle point of the [`Int`] [`Domain`].
/// * `width`: `i64` - Width of the [`Int`] [`Domain`].
///
/// # Examples
///
/// ```
/// use tantale::core::{Int,Domain,DomainBounded};
///
/// let dom = Int::new(0, 10);
///
/// let mut rng = rand::rng();
/// let sampler = dom.default_sampler();
/// assert!(dom.is_in(&sampler(&dom, &mut rng)));
/// assert_eq!(dom.lower(), 0);
/// assert_eq!(dom.upper(), 10);
/// assert_eq!(dom.mid(), 5);
/// assert_eq!(dom.width(), 10);
/// ```
pub type Int = Bounded<i64>;

/// A [`Unit`] domain within `[0,1]`. The floating point type is inferred.
/// /// A generic [`Unit`] [`Domain`] with a numerical `lower=0.0` and `upper=1.0` bounds.
///
pub struct Unit<T>
where
    T: Float + Num + NumCast,
    T: PartialOrd,
    T: Clone + Copy,
    T: Display,
{
    bounds:RangeInclusive<T>
}

impl<T> Unit<T>
where
    T: Float + Num + NumCast,
    T: PartialEq + PartialOrd,
    T: SampleUniform,
    T: Clone + Copy,
    T: Display+Debug,
{
    /// Fabric for a [`Unit`] [`Domain`].
    pub fn new() -> Unit<T> {
        Unit{bounds:RangeInclusive::new(T::from(0.0).unwrap(), T::from(1.0).unwrap())}
    }
}

impl<T> Domain for Unit<T>
where
    T: Float + Num + NumCast,
    T: PartialEq + PartialOrd,
    T: SampleUniform,
    T: Clone + Copy,
    T: Display+Debug,
{
    type TypeDom = T;

    /// Default sampler for [`Unit`].
    /// See [`uniform`].
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        |s, rng| uniform(&s.bounds, rng)
    }

    fn is_in(&self, item: &Self::TypeDom) -> bool {
        self.bounds.contains(item)
    }
}

impl<T> DomainBounded for Unit<T>
where
    T: Float + Num + NumCast,
    T: PartialEq + PartialOrd,
    T: SampleUniform,
    T: Clone + Copy,
    T: Display+Debug,
{
    fn lower(&self) -> Self::TypeDom {
        T::from(0.0).unwrap()
    }
    fn upper(&self) -> Self::TypeDom {
        T::from(1.0).unwrap()
    }
    fn mid(&self) -> Self::TypeDom {
        Self::TypeDom::from(0.5).unwrap()
    }
    fn width(&self) -> Self::TypeDom {
        T::from(1.0).unwrap()
    }
}

impl<T> std::clone::Clone for Unit<T>
where
    T: Float+Num + NumCast,
    T: PartialEq + PartialOrd,
    T: SampleUniform,
    T: Clone + Copy,
    T: Display+Debug,
{
    fn clone(&self) -> Self {
        Unit::new()
    }
}

impl<T> fmt::Display for Unit<T>
where
    T: Float + Num + NumCast,
    T: PartialOrd,
    T: SampleUniform,
    T: Clone + Copy,
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}

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
    /// # Arguments
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
/// let dom = Cat::new(activation);
///
/// let sampler = dom.default_sampler();
/// assert!(dom.is_in(&sampler(&dom, &mut rng)));
/// assert_eq!(dom.values(), check);
/// ```
#[derive(Clone, Copy)]
pub struct Cat<'a, const N: usize> {
    values: [&'a str; N],
}
impl<'a, const N: usize> Cat<'a, N> {
    /// Fabric for a [`Cat`].
    ///
    /// # Arguments
    ///
    ///  * `values` : `[&'a str; N]` - An array of the features defining the categorical [`Domain`].
    ///
    pub fn new(values: [&'a str; N]) -> Cat<'a, N> {
        Cat { values }
    }
    /// Getter for values
    pub fn values(&self) -> [&'a str; N] {
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
    /// # Arguments
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
    /// let cat_1 = Cat::new(activation);
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
