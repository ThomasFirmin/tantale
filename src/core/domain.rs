#[doc(alias = "Domain")]
/// # Domain
/// This crate describes what a domain of a variable is.
/// Most of the domains implements the [`Domain`] type trait [`TypeDom`]. It gives the type of a point within this domain.
/// Domains are use in [`crate::core::variable::Variable`] to define the type of the variable, its `TypeObjective` and `TypeOptimizer`, repectively the input type of that variable within the [`crate::core::objective:Objective`] function, and the input type of the [`crate::core::optimizer::Optimizer`].
use crate::core::errors::DomainError;
use crate::core::sampler::{uniform_bool, uniform_cat, uniform_int, uniform_nat, uniform_real};

use rand::prelude::ThreadRng;
use std::fmt;

/// [`Domain`] is a trait describing the type of a point from the domain it is attached to. It must implement the `default_sampler` method.
#[allow(dead_code)]
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
#[allow(dead_code)]
pub trait NumericallyBounded: Domain {
    /// Getter method for the lower bound.
    fn lower(&self) -> <Self as Domain>::TypeDom;
    /// Getter method for the upper bound.
    fn upper(&self) -> <Self as Domain>::TypeDom;
    /// Getter method for the middle of the axis.
    fn mid(&self) -> <Self as Domain>::TypeDom;
    /// Getter method for the middle of the axis.
    fn range(&self) -> <Self as Domain>::TypeDom;
    /// Getter method for the lower and upper bounds.
    fn bounds(&self) -> (<Self as Domain>::TypeDom, <Self as Domain>::TypeDom) {
        (self.lower(), self.upper())
    }
}

// _-_-_-_-_-_-__-_-_-_-_-_-_-_

// Real domain

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
#[derive(Clone, Copy)]
pub struct Real {
    lower: f64,
    upper: f64,
    mid: f64,
    range: f64,
}
impl Real {
    /// Fabric for the [`Real`] `struct`. It returns a [`Result<Real, DomainError>`].
    ///
    /// The `mid` attribute is automatically computed with $\frac{\texttt{lower}+\texttt{upper}}{2}$.
    ///
    /// # Arguments
    /// * `lower`: `f64` - Lower bound of the [`Real`] domain.
    /// * `upper`: `f64` - Upper bound of the [`Real`] domain.
    /// * `mid`: `f64` - Middle point of the [`Real`] domain. $\frac{\texttt{upper}-\texttt{lower}}{2}$
    /// * `range`: `f64` - Range of the [`Real`] domain.$\texttt{upper}-\texttt{lower}$
    /// 
    /// # Errors
    ///
    /// If `$\texttt{lower} > \texttt{upper}$`, returns a [`DomainError`].
    pub fn new(lower: f64, upper: f64) -> Result<Real, DomainError> {
        if lower < upper {
            let mid = (upper + lower) / 2.0;
            let range = upper - lower;
            Ok(Real { lower, upper, mid,range })
        } else {
            Err(DomainError {
                code: 100,
                msg: String::from(format!("{} is not < {}", lower, upper)),
            })
        }
    }
}
impl Domain for Real {
    type TypeDom = f64;

    /// Default sampler for [`Real`] is a uniform distribution between
    /// `lower` and `upper`.
    /// See [`uniform_real`]
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        uniform_real
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
    /// use tantale::core::{Real, Domain};
    ///
    /// let mut rng = rand::rng();
    /// let real_1 = Real::new(0.0, 10.0).unwrap();
    /// let sampler = real_1.default_sampler();
    /// assert!(real_1.is_in(&sampler(&real_1, &mut rng)));
    /// ```
    ///
    fn is_in(&self, point: &Self::TypeDom) -> bool {
        *point >= self.lower && *point <= self.upper
    }
}

impl NumericallyBounded for Real {
    /// Getter for `lower` attribute of [`Real`].
    fn lower(&self) -> <Self as Domain>::TypeDom {
        self.lower
    }
    /// Getter for `upper` attribute of [`Real`].
    fn upper(&self) -> <Self as Domain>::TypeDom {
        self.upper
    }
    /// Getter for `mid` attribute of [`Real`].
    fn mid(&self) -> <Self as Domain>::TypeDom {
        self.mid
    }
    /// Getter for `range` attribute of [`Real`].
    fn range(&self) -> <Self as Domain>::TypeDom {
        self.range
    }
}

impl fmt::Display for Real {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.lower, self.upper)
    }
}

// _-_-_-_-_-_-__-_-_-_-_-_-_-_

// Naturals domain
/// Describes a [`Domain`] of naturals bounded by a lower and upper bounds.
///
/// # Attributes
///
/// For safety, the attributes are private and can be accessed via getter methods `.lower()`, `.upper()` and `.mid()`.
/// It prevents any modification of the [`Domain`] during the optimization process. Some algorithms might modify the [`Domain`], for example *divide-and-conquer* approaches.
/// In that case, the algorithm should create a new [`Domain`].
///
/// * `lower`: `u64` - Lower bound of the [`Nat`] domain.
/// * `upper`: `u64` - Upper bound of the [`Nat`] domain.
/// * `mid`: `u64` - Middle point of the [`Nat`] domain. $\frac{\texttt{upper}-\texttt{lower}}{2}$
/// * `range`: `u64` - Range of the [`Nat`] domain.$\texttt{upper}-\texttt{lower}$
///
/// # Examples
///
/// ```
/// use tantale::core::{Nat,NumericallyBounded};
/// let natdom = Nat::new(0, 10).unwrap();
/// assert_eq!(natdom.lower(), 0);
/// assert_eq!(natdom.upper(), 10);
/// assert_eq!(natdom.bounds(), (0, 10));
/// ```
#[derive(Clone, Copy)]
pub struct Nat {
    lower: u64,
    upper: u64,
    mid: u64,
    range: u64,
}
impl Nat {
    /// Fabric for the [`Nat`] `struct`. It returns a [`Result<Nat, DomainError>`].
    ///
    /// The `mid` attribute is automatically computed with $\frac{\texttt{lower}+\texttt{upper}}{2}$.
    ///
    /// # Arguments
    ///
    /// * `lower`: `u64` - Lower bound of the [`Nat`] domain.
    /// * `upper`: `u64` - Upper bound of the [`Nat`] domain.
    ///
    /// # Errors
    ///
    /// If the $\texttt{lower} > \texttt{upper}$, returns a [`DomainError`].
    pub fn new(lower: u64, upper: u64) -> Result<Nat, DomainError> {
        let range = std::panic::catch_unwind(|| upper - lower);
        if range.is_err() || *range.as_ref().unwrap() < 2 {
            return Err(DomainError {
                code: 100,
                msg: String::from(format!("{} - {} is not > 1", upper, lower)),
            });
        } else {
            let mid = (upper + lower) / 2;
            Ok(Nat { lower, upper, mid, range:range.unwrap()})
        }
    }
}
impl Domain for Nat {
    type TypeDom = u64;

    /// Default sampler for [`Nat`] is a uniform distribution between
    /// `lower` and `upper`.
    /// See [`uniform_nat`]
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        uniform_nat
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
    /// use tantale::core::{Nat, Domain};
    ///
    /// let mut rng = rand::rng();
    /// let nat_1 = Nat::new(0, 10).unwrap();
    /// let sampler = nat_1.default_sampler();
    /// assert!(nat_1.is_in(&sampler(&nat_1, &mut rng)));
    /// ```
    ///
    fn is_in(&self, point: &Self::TypeDom) -> bool {
        *point >= self.lower && *point <= self.upper
    }
}
impl NumericallyBounded for Nat {
    /// Getter for `lower` attribute of [`Nat`].
    fn lower(&self) -> <Self as Domain>::TypeDom {
        self.lower
    }
    /// Getter for `upper` attribute of [`Nat`].
    fn upper(&self) -> <Self as Domain>::TypeDom {
        self.upper
    }
    /// Getter for `mid` attribute of [`Nat`].
    fn mid(&self) -> <Self as Domain>::TypeDom {
        self.mid
    }
    /// Getter for `mid` attribute of [`Nat`].
    fn range(&self) -> <Self as Domain>::TypeDom {
        self.range
    }
}
impl fmt::Display for Nat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}...{}]*", self.lower, self.upper)
    }
}

// _-_-_-_-_-_-__-_-_-_-_-_-_-_

// Integers domain
/// Describes a [`Domain`] of integers bounded by a lower and upper bounds.
///
/// # Attributes
///
/// * `lower`: `i64` - Lower bound of the [`Int`] domain.
/// * `upper`: `i64` - Upper bound of the [`Int`] domain.
/// * `mid`: `i64` - Middle point of the [`Int`] domain. $\frac{\texttt{upper}-\texttt{lower}}{2}$
/// * `range`: `i64` - Range of the [`Int`] domain.$\texttt{upper}-\texttt{lower}$
///
/// # Examples
///
/// ```
/// use tantale::core::{Int,NumericallyBounded};
/// let intdom = Int::new(0, 10).unwrap();
/// assert_eq!(intdom.lower(), 0);
/// assert_eq!(intdom.upper(), 10);
/// assert_eq!(intdom.bounds(), (0, 10));
/// ```
#[derive(Clone, Copy)]
pub struct Int {
    lower: i64,
    upper: i64,
    mid: i64,
    range:i64,
}
impl Int {
    /// Fabric for the [`Int`] `struct`. It returns a [`Result<Int, DomainError>`].
    ///
    /// The `mid` attribute is automatically computed with $\frac{\texttt{lower}+\texttt{upper}}{2}$.
    ///
    /// # Arguments
    ///
    /// * `lower`: `i64` - Lower bound of the [`Int`] domain.
    /// * `upper`: `i64` - Upper bound of the [`Int`] domain.
    ///
    /// # Errors
    ///
    /// If the $\texttt{lower} > \texttt{upper}$, returns a [`DomainError`].
    pub fn new(lower: i64, upper: i64) -> Result<Int, DomainError> {
        let range = upper-lower;
        if range < 2 {
            return Err(DomainError {
                code: 100,
                msg: String::from(format!("{} - {} is not > 1", upper, lower)),
            });
        } else {
            let mid = (upper + lower) / 2;
            Ok(Int { lower, upper, mid, range })
        }
    }
}
impl Domain for Int {
    type TypeDom = i64;
    /// Default sampler for [`Int`] is a uniform distribution between
    /// `lower` and `upper`.
    /// See [`uniform_int`]
    fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
        uniform_int
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
    /// use tantale::core::{Int, Domain};
    ///
    /// let mut rng = rand::rng();
    /// let int_1 = Int::new(0, 10).unwrap();
    /// let sampler = int_1.default_sampler();
    /// assert!(int_1.is_in(&sampler(&int_1, &mut rng)));
    /// ```
    ///
    fn is_in(&self, point: &Self::TypeDom) -> bool {
        *point >= self.lower && *point <= self.upper
    }
}
impl NumericallyBounded for Int {
    /// Getter for `lower` attribute of [`Int`].
    fn lower(&self) -> <Self as Domain>::TypeDom {
        self.lower
    }
    /// Getter for `upper` attribute of [`Int`].
    fn upper(&self) -> <Self as Domain>::TypeDom {
        self.upper
    }
    /// Getter for `mid` attribute of [`Int`].
    fn mid(&self) -> <Self as Domain>::TypeDom {
        self.mid
    }
    /// Getter for `range` attribute of [`Int`].
    fn range(&self) -> <Self as Domain>::TypeDom {
        self.range
    }
}
impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}...{}]", self.lower, self.upper)
    }
}

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
