#[doc(alias = "Domain")]
/// # Domain
/// This crate describes what a domain of a variable is.
/// Most of the domains implements the [`Domain`] type trait [`TypeDom`]. It gives the type of a point within this domain.
/// Domains are use in [`crate::core::variable::Variable`] to define the type of the variable, its `TypeObjective` and `TypeOptimizer`, repectively the input type of that variable within the [`crate::core::objective:Objective`] function, and the input type of the [`crate::core::optimizer::Optimizer`].
///
use crate::core::errors::DomainError;
use crate::core::sampler::{uniform, uniform_bool, uniform_cat};

use num::{Num, NumCast};
use rand::distr::uniform::SampleUniform;
use rand::prelude::ThreadRng;
use std::fmt::{self, Display};
use std::ops::RangeInclusive;

/// [`Domain`] is a trait describing the type of a point from the domain it is attached to. It must implement the `default_sampler` method.
pub trait Domain {
    type TypeDom;
    /// Associated function to automatically return a default [`crate::core::sampler`] for the domain the trait is implemented.
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom;
    /// Returns `true`` if a given borrowed `point` is in the domain. Otherwise returns `false`.
    ///
    /// # Parameters
    ///
    /// * point : &Self::TypeDom - a borrowed point.
    ///
    fn is_in(&self, point: &Self::TypeDom) -> bool;
}

/// Describes a peculiar trait for some of the domains that are numerically bounded by a `lower` and `upper` bound.
pub trait NumericallyBounded: Domain {
    /// Getter method for the lower bound.
    fn lower(&self) -> Self::TypeDom;
    /// Getter method for the upper bound.
    fn upper(&self) -> Self::TypeDom;
    /// Getter method for the middle of the axis.
    fn mid(&self) -> Self::TypeDom;
    /// Getter method for the middle of the axis.
    fn width(&self) -> Self::TypeDom;
    /// Getter method for the lower and upper bounds.
    fn bounds(&self) -> (Self::TypeDom, Self::TypeDom) {
        (self.lower(), self.upper())
    }
}

pub struct Bounded<T:Num+NumCast> {
    bounds: RangeInclusive<T>,
    mid: T,
    width: T,
}

impl<T> Bounded<T>
where
    T : Num + NumCast,
    T : PartialOrd,
    T: Clone,
    T: Display,
{
    /// Fabric for the [`Bounded`] `struct`. It returns a [`Result<Bounded, DomainError>`].
    ///
    /// The `mid` attribute is automatically computed with $\frac{\texttt{lower}+\texttt{upper}}{2}$.
    ///
    /// # Arguments
    /// * `lower`: `T` - Lower bound of the [`Bounded`] [`Domain`].
    /// * `upper`: `T` - Upper bound of the [`Bounded`] [`Domain`].
    /// * `mid`: `T` - Middle point of the [`Bounded`] [`Domain`]. $\frac{\texttt{upper}-\texttt{lower}}{2}$
    /// * `width`: `T` - Width of the [`Bounded`] [`Domain`]. $\texttt{upper}-\texttt{lower}$
    ///
    /// # Errors
    ///
    /// If `$\texttt{lower} > \texttt{upper}$`, returns a [`DomainError`].
    pub fn new(lower: T, upper: T) -> Result<Bounded<T>, DomainError> {
        if lower < upper {
            let mid = (upper.clone() + lower.clone()) / T::from(2i8).unwrap();
            let width = upper.clone() - lower.clone();
            Ok(Bounded {
                bounds: std::ops::RangeInclusive::new(lower, upper),
                mid,
                width,
            })
        } else {
            Err(DomainError {
                code: 100,
                msg: String::from(format!("{} is not < {}", lower, upper)),
            })
        }
    }
}

impl<T: Num + NumCast + PartialOrd + Clone + SampleUniform> Domain for Bounded<T> {
    type TypeDom = T;

    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        |s, rng| uniform(&s.bounds, rng)
    }

    fn is_in(&self, item: &T) -> bool {
        self.bounds.contains(item)
    }
}

impl<T: Num + NumCast + PartialOrd + Clone + SampleUniform> NumericallyBounded for Bounded<T> {
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

impl<T: Num + NumCast + Display> fmt::Display for Bounded<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.bounds.start(), self.bounds.end())
    }
}

/// Describes a continuous [`Domain`] bounded by a lower and upper bounds.
///
/// # Attributes
///
/// For safety, the attributes are private and can be accessed via getter methods `.lower()`, `.upper()` and `.mid()`.
/// It prevents any modification of the [`Domain`] during the optimization process. Some algorithms might modify the [`Domain`], for example *divide-and-conquer* approaches.
/// In that case, the algorithm should create a new [`Domain`].
///
/// * `lower`: `f64` - Lower bound of the [`Real`] domain.
/// * `upper`: `f64` - Upper bound of the [`Real`] domain.
/// * `mid`: `f64` - Middle point of the [`Real`] domain.
///
/// # Examples
///
/// ```
/// use tantale::core::{Real,NumericallyBounded};
/// let realdom = Real::new(0.0, 10.0).unwrap();
/// assert_eq!(realdom.lower(), 0.0);
/// assert_eq!(realdom.upper(), 10.0);
/// assert_eq!(realdom.bounds(), (0.0, 10.0));
/// ```
pub type Real = Bounded<f64>;

pub type Nat = Bounded<u64>;

pub type Int = Bounded<i64>;

// _-_-_-_-_-_-__-_-_-_-_-_-_-_

// Booleans domain
/// Describes a boolean domain .
///
/// # Examples
///
/// ```
/// use tantale::core::Bool;
/// let booldom = Bool::new().unwrap();
/// assert_eq!(booldom.values(),(true, false));
/// ```
#[derive(Clone, Copy)]
pub struct Bool;
impl Bool {
    /// Fabric for the [`Bool`] `struct`. It returns a [`Result<Bool, DomainError>`].
    ///
    /// # Errors
    ///
    /// Should not return a [`DomainError`].
    /// The function returns a [`Result`] for consistency.
    pub fn new() -> Result<Bool, DomainError> {
        Ok(Bool {})
    }
    pub fn values(&self) -> (bool, bool) {
        (true, false)
    }
    pub fn bounds(&self) -> (bool, bool) {
        self.values()
    }
}
impl Domain for Bool {
    type TypeDom = bool;

    /// Default sampler for [`Bool`].
    /// See [`uniform_bool`]
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        uniform_bool
    }
    /// Method to check if a given point is in the domain.
    ///
    /// # Arguments
    ///
    /// * point : `Self::TypeDom` : a point of the same type as the type of the domain.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Bool, Domain};
    ///
    /// let mut rng = rand::rng();
    /// let bool_1 = Bool::new().unwrap();
    /// let sampler = bool_1.default_sampler();
    /// assert!(bool_1.is_in(&sampler(&bool_1, &mut rng)));
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

/// Describes a non-ordinal categorical domain. It is made of features, described by the private attribute `values`, an [`array`] of strings.
/// Each elements describing a unique feature.
/// The values can be accessed by the corresponding getter `values()`.
///
/// # Attributes
///
///  * values : [&'a str; N] - An array of the different features defining the categorical [`Domain`].
///
/// # Examples
///
/// ```
/// use tantale::core::Cat;
/// let activation = ["relu", "tanh", "sigmoid"];
/// let check = ["relu", "tanh", "sigmoid"];
/// let cat_1 = Cat::new(activation).unwrap();
/// assert_eq!(cat_1.values(), check);
/// ```
#[derive(Clone, Copy)]
pub struct Cat<'a, const N: usize> {
    values: [&'a str; N],
}
impl<'a, const N: usize> Cat<'a, N> {
    /// Fabric for the [`Cat`] `struct`. It returns a [`Result<Cat, DomainError>`].
    ///
    /// # Arguments
    ///
    ///  * values : [&'a str; N] - An array of the different features defining the categorical [`Domain`].
    ///
    /// # Errors
    ///
    /// Should not return a [`DomainError`].
    /// The function returns a [`Result`] for consistency.
    pub fn new(values: [&'a str; N]) -> Result<Cat<'a, N>, DomainError> {
        Ok(Cat { values })
    }
    /// Getter for values
    pub fn values(&self) -> [&'a str; N] {
        self.values
    }
    /// Use for consistency with the other domains. It returns the `values``.
    pub fn bounds(&self) -> [&'a str; N] {
        self.values()
    }
}
impl<'a, const N: usize> Domain for Cat<'a, N> {
    /// The type of a point within the domain is a `&'a str`, i.e. a pointer to a `str` from the `values`.
    type TypeDom = &'a str;

    /// Default sampler for [`Cat`] is a uniform choice within the `values`
    /// See [`uniform_cat`]
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        uniform_cat
    }

    /// Method to check if a given point is in the domain.
    ///
    /// # Arguments
    ///
    /// * point : `Self::TypeDom` : a point of the same type as the type of the domain.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Cat,Domain};
    ///
    /// let mut rng = rand::rng();
    /// let activation = ["relu", "tanh", "sigmoid"];
    /// let cat_1 = Cat::new(activation).unwrap();
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

// _-_-_-_-_-_-__-_-_-_-_-_-_-_

// pub struct Composite<const N: usize> {
//     pub array: [Box<dyn Basic>; N],
// }

// impl<const N: usize> fmt::Display for Composite<N> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         let asize = self.array.len() - 1;
//         write!(f, "Composite{{")?;
//         for elem in self.array[..asize].iter() {
//             write!(f, "{}, ", *elem)?;
//         }
//         write!(f, "{}}}", self.array[asize])
//     }
// }
// // impl VariableType for Composite{
// //     type VarType = ;
// // }
