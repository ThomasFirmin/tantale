use crate::core::domain::{Bool, Cat, Domain, DomainBounded, Int, Nat, Real};

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
/// Uniformly sample a real within \[lower,upper\].
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
where
    Real: Domain,
{
    rng.random_range(domain.lower()..domain.upper())
}

/// Discrete uniform ditribution for [`Nat`] [`Domain`].
/// Uniformly sample a real within \[lower,upper\].
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
where
    Nat: Domain,
{
    rng.random_range(domain.lower()..domain.upper())
}

/// Discrete uniform ditribution for [`Int`] [`Domain`].
/// Uniformly sample a real within \[lower,upper\].
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
where
    Int: Domain,
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
where
    Bool: Domain,
{
    rng.random_bool(0.5)
}

fn _from_str_to_typedom<'a, const N: usize>(
    _d: &Cat<'a, N>,
    to_cast: &'a str,
) -> <Cat<'a, N> as Domain>::TypeDom {
    to_cast
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
pub fn uniform_cat<'a, const N: usize>(
    domain: &Cat<'a, N>,
    rng: &mut ThreadRng,
) -> <Cat<'a, N> as Domain>::TypeDom {
    domain.values().iter().choose(rng).unwrap()
}
