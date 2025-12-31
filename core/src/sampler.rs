use crate::domain::{
    bounded::{BoundedBounds, RangeDomain},
    Bool, Bounded, Cat, Domain, TypeDom,
};

use rand::prelude::{IteratorRandom, Rng, ThreadRng};

/// Describes what a [`Sampler`] such as [`Uniform`] or [`Bernoulli`] is.
///
/// # Parameters
///
/// * `rng` : `&mut`[`ThreadRng`](rand::prelude::ThreadRng) - The RNG from [`rand`].
pub trait Sampler<D: Domain> {
    fn sample(&self, dom: &D, rng: &mut ThreadRng) -> D::TypeDom;
}

/// An enum of the different available samplers for [`Bounded`].
#[derive(Clone, Copy, Debug)]
pub enum BoundedDistribution {
    Uniform(Uniform),
}

impl<D> Sampler<D> for BoundedDistribution
where
    D: RangeDomain,
    D::TypeDom: BoundedBounds,
{
    fn sample(&self, dom: &D, rng: &mut ThreadRng) -> D::TypeDom {
        match self {
            BoundedDistribution::Uniform(u) => u.sample(dom, rng),
        }
    }
}

/// An enum of the different available samplers for [`Bool`].
#[derive(Clone, Copy, Debug)]
pub enum BoolDistribution {
    Bernoulli(Bernoulli),
}

impl Sampler<Bool> for BoolDistribution {
    fn sample(&self, dom: &Bool, rng: &mut ThreadRng) -> TypeDom<Bool> {
        match self {
            BoolDistribution::Bernoulli(b) => b.sample(dom, rng),
        }
    }
}

/// An enum of the different available samplers for [`Cat`].
#[derive(Clone, Copy, Debug)]
pub enum CatDistribution {
    Uniform(Uniform),
}

impl Sampler<Cat> for CatDistribution {
    fn sample(&self, dom: &Cat, rng: &mut ThreadRng) -> TypeDom<Cat> {
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

/// Uniformly sample within a [`Bounded`] [`Domain`].
/// This is the default sampler for [`Bounded`] [`Domain`].
///
/// $$\sim \mathcal{U}\[\texttt{lower},\texttt{upper}\]$$
#[derive(Clone, Copy, Debug)]
pub struct Uniform;
impl<D: RangeDomain> Sampler<D> for Uniform
where
    D::TypeDom: BoundedBounds,
{
    fn sample(&self, dom: &D, rng: &mut ThreadRng) -> TypeDom<Bounded<D::TypeDom>> {
        rng.random_range(dom.get_bounds())
    }
}

/// Random choice for a [`Cat`] [`Domain`].
/// Uniformly sample a feature from [`Cat`]'s `values`.
impl Sampler<Cat> for Uniform {
    fn sample(&self, dom: &Cat, rng: &mut ThreadRng) -> TypeDom<Cat> {
        dom.values.iter().choose(rng).unwrap().clone()
    }
}

/// Sample `bool` according to Bernoulli distribution.
/// This is the default sampler for [`Bool`] [`Domain`].
///
/// $$\sim \mathcal{B}(p)$$
#[derive(Clone, Copy, Debug)]
pub struct Bernoulli(pub f64);
impl Sampler<Bool> for Bernoulli {
    fn sample(&self, _dom: &Bool, rng: &mut ThreadRng) -> TypeDom<Bool> {
        rng.random_bool(self.0)
    }
}
