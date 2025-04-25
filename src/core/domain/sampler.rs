use crate::core::domain::{Bool, Cat, Domain, DomainBounded, Int, Nat, Real,Unit};

use rand::{
    distr::uniform::SampleUniform,
    prelude::{IteratorRandom, Rng, ThreadRng},
};

/// Uniform distribution.
/// Uniformly sample within a [`DomainBounded`] [`Domain`].
/// This is the default sampler for [`DomainBounded`] [`Domain`].
///
/// $$\sim \mathcal{U}\[\texttt{lower},\texttt{upper}\]$$
///
/// # Arguments
///
/// * `domain` : `&`[`DomainBounded`] - A borrowed [`DomainBounded`] [`Domain`].
/// * rng : `&mut `[`ThreadRng`] - A mutable reference to a thread-local generator.
///
pub fn uniform<T: PartialOrd + SampleUniform + Clone>(
    domain: &std::ops::RangeInclusive<T>,
    rng: &mut ThreadRng,
) -> T {
    rng.random_range(domain.clone())
}

/// Uniform distribution for [`Real`] [`Domain`].
/// Uniformly sample a real within $[\texttt{lower},\texttt{upper}]$.
/// This is the default sampler for [`Real`].
///
/// $$\sim \mathcal{U}\[\texttt{lower},\texttt{upper}\]$$
///
/// # Arguments
///
/// * `domain` : `&`[`Real`] - A borrowed [`Real`] [`Domain`].
/// * rng : `&mut `[`ThreadRng`] - A mutable reference to a thread-local generator.
///
pub fn uniform_real(domain: &Real, rng: &mut ThreadRng) -> <Real as Domain>::TypeDom
{
    rng.random_range(domain.lower()..domain.upper())
}

/// Discrete uniform ditribution for [`Nat`] [`Domain`].
/// Uniformly sample a real within $[\texttt{lower},\texttt{upper}]$.
/// This is the default sampler for [`Nat`].
///
/// $$\sim \mathcal{U}\\{\texttt{lower},\texttt{upper}\\}$$
///
/// # Arguments
///
/// * `domain` : `&`[`Nat`] - A borrowed [`Nat`] [`Domain`].
/// * rng : `&mut `[`ThreadRng`] - A mutable reference to a thread-local generator.
///
pub fn uniform_nat(domain: &Nat, rng: &mut ThreadRng) -> <Nat as Domain>::TypeDom
{
    rng.random_range(domain.lower()..domain.upper())
}

/// Discrete uniform ditribution for [`Int`] [`Domain`].
/// Uniformly sample a real within $[\texttt{lower},\texttt{upper}]$.
/// This is the default sampler for [`Int`].
///
/// $$\sim \mathcal{U}\\{\texttt{lower},\texttt{upper}\\}$$
///
/// # Arguments
///
/// * `domain` : `&`[`Int`] - A borrowed [`Int`] [`Domain`].
/// * rng : `&mut `[`ThreadRng`] - A mutable reference to a thread-local generator.
///
pub fn uniform_int(domain: &Int, rng: &mut ThreadRng) -> <Int as Domain>::TypeDom
{
    rng.random_range(domain.lower()..domain.upper())
}
/// Sample a bool with 50% of chance of being `true`.
/// This is the default sampler for [`Bool`].
///
/// # Arguments
///
/// * `domain` : `&`[`Bool`] - A borrowed [`Bool`] [`Domain`].
/// * rng : `&mut `[`ThreadRng`] - A mutable reference to a thread-local generator.
///
pub fn uniform_bool(_domain: &Bool, rng: &mut ThreadRng) -> <Bool as Domain>::TypeDom
{
    rng.random_bool(0.5)
}

/// Random choice for [`Cat`] [`Domain`].
/// Uniformly sample a feature from [`Cat`]'s `values`.
/// This is the default sampler for [`Cat`].
///
/// # Arguments
///
/// * `domain` : `&`[`Cat`] - A borrowed [`Cat`] [`Domain`].
/// * rng : `&mut `[`ThreadRng`] - A mutable reference to a thread-local generator.
///
pub fn uniform_cat<'a>(
    domain: &Cat<'a>,
    rng: &mut ThreadRng,
) -> <Cat<'a> as Domain>::TypeDom {
    domain.values().iter().choose(rng).unwrap()
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
pub fn uniform_unit(_domain: &Unit, rng: &mut ThreadRng) -> <Unit as Domain>::TypeDom
{
    rng.random()
}