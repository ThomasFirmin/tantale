use crate::{domain::{Bool, Bounded, Cat, Domain, TypeDom, Unit, bounded::BoundedBounds}};

use rand::prelude::{IteratorRandom, Rng, ThreadRng};


pub trait Sampler<Dom:Domain>{
    fn sample(&mut self, dom:&Dom, rng:&mut ThreadRng)->Dom::TypeDom;
}

/// Uniformly sample within a [`Bounded`] [`Domain`].
/// This is the default sampler for [`Bounded`] [`Domain`].
///
/// $$\sim \mathcal{U}\[\texttt{lower},\texttt{upper}\]$$
pub struct Uniform;
impl<T: BoundedBounds> Sampler<Bounded<T>> for Uniform
{
    fn sample(&mut self, dom:&Bounded<T>, rng:&mut ThreadRng)->TypeDom<Bounded<T>> {
        rng.random_range(dom.bounds)
    }
}

/// Random choice for a [`Cat`] [`Domain`].
/// Uniformly sample a feature from [`Cat`]'s `values`.
impl Sampler<Cat> for Uniform
{
    fn sample(&mut self, dom:&Cat, rng:&mut ThreadRng)->TypeDom<Cat> {
        dom.values.iter().choose(rng).unwrap().clone()
    }
}

/// Sample `bool` according to Bernoulli distribution.
/// This is the default sampler for [`Bool`] [`Domain`].
///
/// $$\sim \mathcal{B}(p)$$
pub struct Bernoulli(f64);
impl Sampler<Bool> for Bernoulli{
    fn sample(&mut self, dom:&Bool, rng:&mut ThreadRng)-><Bool as Domain>::TypeDom {
        rng.random_bool(self.0)
    }
}

/// Uniform distribution for [`Unit`] [`Domain`].
/// Uniformly sample a real within $[0.0,1.0]$.
/// This is the default sampler for [`Unit`].
///
/// $$\sim \mathcal{U}\[\texttt{lower},\texttt{upper}\]$$
///
/// # Arguments
///
/// * `domain` : `&`[`Unit`] - A borrowed [`Unit`] [`Domain`].
/// * rng : `&mut `[`ThreadRng`] - A mutable reference to a thread-local generator.
///
pub fn uniform_unit(_domain: &Unit, rng: &mut ThreadRng) -> TypeDom<Unit> {
    rng.random()
}
