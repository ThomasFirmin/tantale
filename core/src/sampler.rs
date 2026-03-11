//! # Sampler
//!
//! This module provides sampling strategies for different [`Domain`] types. Samplers define how
//! random values are drawn from a [`Domain`].
//!
//! ## Overview
//!
//! The [`Sampler`] trait abstracts the process of generating random values from a [`Domain`].
//! Different domains support different sampling strategies:
//! - Bounded numeric domains: [`Uniform`] sampling
//! - Boolean domains: [`Bernoulli`] sampling
//! - Categorical domains: [`Uniform`] random choice
//!
//! ## Distribution Enums
//!
//! For flexibility, Tantale provides enum wrappers for domain-specific samplers:
//! - [`BoundedDistribution`] - Samplers for [`Bounded`] numeric domains
//! - [`BoolDistribution`] - Samplers for [`Bool`] domains
//! - [`CatDistribution`] - Samplers for [`Cat`] categorical domains
//!
//! These enums allow runtime selection of sampling strategies when needed.
//!
//! ## Some built-in samplers
//!
//! ### [`Uniform`]provides a runtime-selectable wrapper
//! - **For numeric domains**: Samples uniformly over `[lower, upper]`
//! - **For categorical domains**: Random choice among features with equal probability
//!
//! ### [`Bernoulli`]
//! - **For boolean domains**: Bernoulli distribution with configurable probability $p$
//!
//! ## Usage
//!
//! ```
//! use tantale::core::{Domain, Real, Sampler, Uniform};
//!
//! let domain = Real::new(0.0, 1.0, Uniform);
//! let mut rng = rand::rng();
//! let value = domain.sample(&mut rng);
//! ```
//!
//! ## See Also
//!
//! - [`Domain::sample`] - Default sampling method for domains
//! - [`Domain`] - Core domain trait
//! - [`Bounded`] - Numeric range domains
//! - [`Bool`] - Boolean domain
//! - [`Cat`] - Categorical domain

use crate::domain::{
    Bool, Bounded, Cat, Domain, TypeDom,
    bounded::{BoundedBounds, RangeDomain},
};

use rand::prelude::{IteratorRandom, Rng};

/// Core trait for sampling strategies that generate random values from a domain.
///
/// A [`Sampler`] defines how to draw random values from a given [`Domain`]. Different samplers
/// implement different probability distributions or selection strategies. This abstraction allows
/// optimizers and initialization routines to use customizable sampling approaches.
///
/// # Type Parameter
///
/// * `D` - The domain type from which values are sampled
///
/// # Implementations
///
/// Tantale provides built-in samplers:
/// - [`Uniform`] - Uniform distribution for numeric and categorical domains
/// - [`Bernoulli`] - Bernoulli distribution for boolean domains
///
/// # Example
///
/// ```
/// use tantale::core::{Domain, Real, Sampler, Uniform};
///
/// let domain = Real::new(0.0, 10.0, Uniform);
/// let mut rng = rand::rng();
///
/// // Sample a random value from [0.0, 10.0]
/// let value = domain.sample(&mut rng);
/// assert!(domain.is_in(&value));
/// ```
pub trait Sampler<D: Domain> {
    /// Generates a random value from the given domain using this sampling strategy.
    ///
    /// # Parameters
    ///
    /// * `dom` - The domain from which to sample
    /// * `rng` - A random number generator implementing [`Rng`]
    ///
    /// # Returns
    ///
    /// A randomly sampled value of type [`D::TypeDom`](Domain::TypeDom) drawn according
    /// to this sampler's distribution.
    fn sample<R: Rng>(&self, dom: &D, rng: &mut R) -> D::TypeDom;
}

/// Enumeration of available sampling distributions for bounded numeric domains.
///
/// [`BoundedDistribution`] give a wrapper around different sampling strategies applicable to [`Bounded`] domains
/// (e.g., [`Real`](crate::domain::Real),[`Int`](crate::domain::Int), [`Nat`](crate::domain::Nat)).
///
/// # Variants
///
/// - [`Uniform`](BoundedDistribution::Uniform) - Uniform distribution over the domain bounds
/// - Can be extended in future works (e.g. Gaussian)...
///
/// # Conversion
///
/// Individual samplers can be converted into this enum via [`From`]/[`Into`]:
///
/// ```
/// use tantale::core::{Uniform,BoundedDistribution};
///
/// let dist: BoundedDistribution = Uniform.into();
/// ```
#[derive(Clone, Copy, Debug)]
pub enum BoundedDistribution {
    Uniform(Uniform),
}

impl<D> Sampler<D> for BoundedDistribution
where
    D: RangeDomain,
    D::TypeDom: BoundedBounds,
{
    fn sample<R: Rng>(&self, dom: &D, rng: &mut R) -> D::TypeDom {
        match self {
            BoundedDistribution::Uniform(u) => u.sample(dom, rng),
        }
    }
}

/// Enumeration of available sampling distributions for boolean domains.
///
/// [`BoolDistribution`] provides a wrapper around different sampling
/// strategies applicable to [`Bool`] domains.
///
/// # Variants
///
/// - [`Bernoulli`](BoolDistribution::Bernoulli) - Bernoulli distribution with configurable probability
/// - Can be extended in future works...
///
/// # Conversion
///
/// Individual samplers can be converted into this enum via [`From`]/[`Into`]:
///
/// ```
/// use tantale::core::{Bernoulli,BoolDistribution};
///
/// let dist: BoolDistribution = Bernoulli(0.5).into();
/// ```
#[derive(Clone, Copy, Debug)]
pub enum BoolDistribution {
    Bernoulli(Bernoulli),
}

impl Sampler<Bool> for BoolDistribution {
    fn sample<R: Rng>(&self, dom: &Bool, rng: &mut R) -> TypeDom<Bool> {
        match self {
            BoolDistribution::Bernoulli(b) => b.sample(dom, rng),
        }
    }
}

/// Enumeration of available sampling distributions for categorical domains.
///
/// [`CatDistribution`] provides a wrapper around different sampling
/// strategies applicable to [`Cat`] categorical domains.
///
/// # Variants
///
/// - [`Uniform`](CatDistribution::Uniform) - Uniform random selection among categories
/// - Can be extended in future works...
///
/// # Conversion
///
/// Individual samplers can be converted into this enum via [`From`]/[`Into`]:
///
/// ```
/// use tantale::core::{CatDistribution,Uniform};
///
/// let dist: CatDistribution = Uniform.into();
/// ```
#[derive(Clone, Copy, Debug)]
pub enum CatDistribution {
    Uniform(Uniform),
}

impl Sampler<Cat> for CatDistribution {
    fn sample<R: Rng>(&self, dom: &Cat, rng: &mut R) -> TypeDom<Cat> {
        match self {
            CatDistribution::Uniform(u) => u.sample(dom, rng),
        }
    }
}

impl From<Uniform> for BoundedDistribution {
    fn from(val: Uniform) -> Self {
        BoundedDistribution::Uniform(val)
    }
}

impl From<Uniform> for CatDistribution {
    fn from(val: Uniform) -> Self {
        CatDistribution::Uniform(val)
    }
}

impl From<Bernoulli> for BoolDistribution {
    fn from(val: Bernoulli) -> Self {
        BoolDistribution::Bernoulli(val)
    }
}

/// Uniform distribution sampler for bounded and categorical domains.
///
/// [`Uniform`] samples values with equal probability across the entire domain:
/// - For bounded numeric domains ([`Real`](crate::domain::Real), [`Int`](crate::domain::Int),
///   [`Nat`](crate::domain::Nat)): samples uniformly over $[\texttt{lower}, \texttt{upper}]$
/// - For categorical domains ([`Cat`]): selects uniformly among features
///
/// # Mathematical Definition
///
/// For bounded domains:
/// $$X \sim \mathcal{U}[\texttt{lower}, \texttt{upper}]$$
///
/// For categorical domains with $n$ features:
/// $$P(X = c_i) = \frac{1}{n} \quad \forall i \in \{1, \ldots, n\}$$
///
/// # Usage
///
/// This is the default sampler for bounded and categorical domains.
///
/// ```
/// use tantale::core::{Sampler, Uniform, Real};
///
/// let domain = Real::new(0.0, 1.0, Uniform);
/// let mut rng = rand::rng();
///
/// // Sample uniformly from [0.0, 1.0]
/// let value = Uniform.sample(&domain, &mut rng);
/// assert!(value >= 0.0 && value <= 1.0);
/// ```
///
/// # See Also
///
/// - [`Bernoulli`] - Alternative sampler for boolean domains
/// - [`BoundedDistribution`] - Enum wrapper for bounded domain samplers
/// - [`CatDistribution`] - Enum wrapper for categorical domain samplers
#[derive(Clone, Copy, Debug)]
pub struct Uniform;
impl<D: RangeDomain> Sampler<D> for Uniform
where
    D::TypeDom: BoundedBounds,
{
    fn sample<R: Rng>(&self, dom: &D, rng: &mut R) -> TypeDom<Bounded<D::TypeDom>> {
        rng.random_range(dom.get_bounds())
    }
}

/// Uniform sampler implementation for categorical domains.
///
/// Selects a random category from the domain's values with equal probability.
impl Sampler<Cat> for Uniform {
    fn sample<R: Rng>(&self, dom: &Cat, rng: &mut R) -> TypeDom<Cat> {
        dom.values.iter().choose(rng).unwrap().clone()
    }
}

/// Bernoulli distribution sampler for boolean domains.
///
/// [`Bernoulli`] samples boolean values according to a Bernoulli distribution with
/// parameter $p \in [0, 1]$, where $p$ is the probability of sampling `true`.
///
/// # Mathematical Definition
///
/// $$X \sim \mathcal{B}(p)$$
///
/// Where:
/// - $P(X = \texttt{true}) = p$
/// - $P(X = \texttt{false}) = 1 - p$
///
/// # Constructor
///
/// The single field `f64` value is the probability parameter $p$.
///
/// ```ignore
/// use tantale::core::{Sampler, Bernoulli, Bool};
/// use rand::thread_rng;
///
/// let domain = Bool(Bernoulli(0.7));
/// let mut rng = thread_rng();
///
/// // Sample from Bernoulli(0.7)
/// let value = bernoulli.sample(&domain, &mut rng);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Bernoulli(pub f64);
impl Sampler<Bool> for Bernoulli {
    fn sample<R: Rng>(&self, _dom: &Bool, rng: &mut R) -> TypeDom<Bool> {
        rng.random_bool(self.0)
    }
}
