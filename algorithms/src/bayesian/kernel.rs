use tantale_core::{
    Bool, Domain, GridDom, HasVariables, HasX, Id, Int, Mixed, MixedTypeDom, Nat, Outcome, Real,
    Searchspace, SolInfo, Uncomputed, Unit, Xy,
    domain::{CategoricalDomain, NumericalDomain, TypeDom, codomain::TypeCodom, grid::GridBounds},
};

use core::f64;
use num::{Num, cast::AsPrimitive};
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use statrs::function::erf;
use std::sync::Arc;

use crate::bayesian::{
    bandwidth::BandwidthType, weighter::PointWeights,
};

const SQRT_2PI: f64 = 2.5066282746310002;

/// Computes :
/// $$
/// g(x, mean \\,\\lvert\\, std) = \\frac{1}{\\sqrt{2 \\pi std^2}}\\exp\\left( -\\left(\\frac{(x - mean)^2}{2 std^2}\\right) \\right)\\enspace\\text{,}
/// $$
/// where $x$ is the point at which to evaluate the PDF, `mean` is the mean of the Gaussian distribution, and `std` is the standard deviation of the Gaussian distribution.
pub fn gaussian_pdf(mean: f64, std: f64, x: f64) -> f64 {
    let coeff = 1.0 / (std * SQRT_2PI);
    let exponent = -0.5 * ((x - mean) / std).powi(2);
    coeff * exponent.exp()
}

pub fn gaussian_cdf<T: Num + AsPrimitive<f64>>(mean: f64, std: f64, x: &T) -> f64 {
    0.5 * erf::erfc((mean - x.as_()) / (std * f64::consts::SQRT_2))
}

pub fn gaussian_icdf<T: Num + AsPrimitive<f64>>(mean: f64, std: f64, x: &T) -> f64 {
    mean - (std * f64::consts::SQRT_2 * erf::erfc_inv(2.0 * x.as_()))
}

/// Computes :
/// $$
/// \\begin{split}
///     \\mathbb{P}\\left( L < X = x_2 < U \\right) &= \\int_{L}^{U} K(x,x_2\\,|\\,b)dx \\\\
///                 &= \\frac{1}{2}\\left(\\text{erf}(\\frac{(R - x_2)}{(\\sqrt{2} b)}) -\\text{erf}(\\frac{(L - x_2)}{(\\sqrt{2} b)}) \\right)
/// \\end{split}
/// $$
fn gaussian_interval<T: Num + AsPrimitive<f64>>(x: &T, bandwidth: f64, low: f64, up: f64) -> f64 {
    let x_f = x.as_();
    let denom = f64::consts::SQRT_2 * bandwidth;
    let low_erf = erf::erf((low - x_f) / denom);
    let up_erf = erf::erf((up - x_f) / denom);
    (up_erf - low_erf) / 2.0
}

pub trait KernelFunc<Dom: Domain> {
    /// The context type for the kernel,
    /// which can hold precomputed values or parameters needed
    /// for efficient kernel computation for each point within the archive.
    type Context: Serialize + for<'a> Deserialize<'a>;

    fn get_context(x: &Dom::TypeDom, bandwidth: f64, dom: &Dom) -> Self::Context;

    /// Computes the kernel function between two [`SolutionShape`](tantale_core::SolutionShape) instances of `Opt` type `Dom`.
    ///
    /// # Arguments
    /// * `x1` - The first point.
    /// * `x2` - The second point.
    /// * `context` - The context for the kernel computation.
    /// * `dom` - The domain of the input point.
    ///
    /// # Returns
    /// The kernel value between the two [`Dom::TypeDom`](tantale_core::Domain::TypeDom).
    fn compute(
        x1: &Dom::TypeDom,
        x2: &Dom::TypeDom,
        context: &Self::Context,
        dom: &Dom,
    ) -> f64;

    /// Computes the prior probability for a given point `x` in the domain `dom`.
    fn prior(x: &Dom::TypeDom, dom: &Dom) -> f64;

    /// Samples a value from the kernel distribution at `x` with the given `bandwidth` and domain `dom`.
    fn sample<R: Rng>(
        rng: &mut R,
        x: &Dom::TypeDom,
        context: &Self::Context,
        dom: &Dom,
    ) -> Dom::TypeDom;
}

/// The truncated Gaussian kernel for a single element of a bounded numerical solution is given by the following equation:
/// $$
/// K(x_1, x_2 \\,\\lvert\\, b) = \\frac{g(x_1, x_2 \\,\\lvert\\, b)}{Z(x_2 \\,\\lvert\\, b)}\\enspace\\text{,}
/// $$
/// with the Gaussian kernel:
/// $$
/// g(x_1, x_2 \\,\\lvert\\, b) = \\frac{1}{\\sqrt{2 \\pi b^2}}\\exp\\left( -\\frac{1}{2}\\left(\\frac{x_1 - x_2}{b}\\right)^2 \\right)\\enspace\\text{,}
/// $$
/// where $b$ is the bandwidth parameter that controls the smoothness of the kernel.
/// A smaller $b$ results in a more localized kernel, while a larger $b$ results in a smoother kernel.
/// The normalization is constant computed as:
/// $$
/// \\begin{aligned}
///     Z(x_2 \\,\\lvert\\, b) &= \\int_{L}^{U} K(x,x_2\\,|\\,b)\\,dx \\\\
///            &= \\frac{1}{2}\\left(
///               \\operatorname{erf}\\left(
///                   \\frac{U - x_2}{\\sqrt{2}\\,b}
///               \\right)
///               -
///               \\operatorname{erf}\\left(
///                   \\frac{L - x_2}{\\sqrt{2}\\,b}
///               \\right)
///            \\right)
/// \\end{aligned}\\enspace\\text{.}
/// $$
/// For [`Int`] and [`Nat`] domains defined as $[L, L+1, \ldots, U]$, the kernel function is computed as:
/// $$
///     K^\\prime(x_1, x_2 \\,\\lvert\\, b) = \\frac{1}{Z^\\prime(x_2 \\,\\lvert\\, b)}\\int_{x_1 - \frac{1}{2}}^{x_1 + \frac{1}{2}} g(x,x_2 \\,\\lvert\\, b)dx \\enspace\\text{,}
/// $$
/// with $Z^\\prime(x_2 \\,\\lvert\\, b) = \\int_{L-\frac{1}{2}}^{U+\frac{1}{2}} g(x,x_2 \\,\\lvert\\, b)dx$.
/// Conversely to the [`Real`] case, a continuity correction is applied to for $Z^\\prime$, which explains $\\int_{L-\frac{1}{2}}^{U+\frac{1}{2}}$.
///
/// # Arguments
///
/// * `bandwidth` - The bandwidth parameter of the Gaussian kernel.
/// * `lhs` - (private) Save the left-hand side constant for efficiency, computed as $\frac{1}{\sqrt{2\pi b^2}}$.
pub struct GaussianKernel;

#[derive(Serialize, Deserialize)]
pub struct GaussianContext {
    pub bandwidth: f64,
    pub lhs: f64,
    pub cst: f64,
    pub p_low: f64,
    pub p_up: f64,
}

impl KernelFunc<Real> for GaussianKernel {
    
    type Context = GaussianContext;

    fn get_context(x: &f64, bandwidth: f64, dom: &Real) -> Self::Context {
        let lhs = 1. / (bandwidth * SQRT_2PI);
        let (low, up) = dom.get_bounds();
        let cst = gaussian_interval(x, bandwidth, low, up);
        let p_low = gaussian_cdf(*x, bandwidth, &low);
        let p_up = gaussian_cdf(*x, bandwidth, &up);
        GaussianContext { bandwidth, lhs, cst, p_low, p_up }
    }

    fn compute(
        x1: &f64,
        x2: &f64,
        context: &Self::Context,
        _dom: &Real,
    ) -> f64 {
        context.lhs * (-0.5 * ((x1 - x2) / context.bandwidth).powi(2)).exp() / context.cst
    }

    fn prior(x: &f64, dom: &Real) -> f64 {
        let (low, up) = dom.get_bounds();
        let mean = (low + up) / 2.0;
        let std = up - low;
        // Compute Gaussian PDF
        gaussian_pdf(mean, std, *x)
    }

    /// Samples a value from the truncated Gaussian distribution at `x` with the given
    /// `bandwidth` and domain `dom`.
    ///
    /// # Notes
    ///
    /// The sampling is performed using the inverse transform sampling method,
    /// which involves sampling a uniform random variable and applying the inverse CDF
    /// of the truncated Gaussian distribution.
    fn sample<R: Rng>(
        rng: &mut R,
        x: &f64,
        context: &Self::Context,
        dom: &Real,
    ) -> f64 {
        let (low, up) = dom.get_bounds();
        if (context.p_up - context.p_low).abs() < f64::EPSILON {
            return *x;
        }
        let u: f64 = rng.random_range(context.p_low..context.p_up);
        gaussian_icdf(*x, context.bandwidth, &u).clamp(low, up)
    }
}

impl KernelFunc<Unit> for GaussianKernel {
    type Context = GaussianContext;

    fn get_context(x: &f64, bandwidth: f64, _dom: &Unit) -> Self::Context {
        let lhs = 1. / (bandwidth * SQRT_2PI);
        let cst = gaussian_interval(x, bandwidth, 0.0, 1.0);
        let p_low = gaussian_cdf(*x, bandwidth, &0.0);
        let p_up = gaussian_cdf(*x, bandwidth, &1.0);
        GaussianContext { bandwidth, lhs, cst, p_low, p_up }
    }

    fn compute(
        x1: &f64,
        x2: &f64,
        context: &Self::Context,
        _dom: &Unit,
    ) -> f64 {
        context.lhs * (-0.5 * ((x1 - x2) / context.bandwidth).powi(2)).exp() / context.cst
    }

    fn prior(x: &f64, _dom: &Unit) -> f64 {
        gaussian_pdf(0.5, 1.0, *x)
    }

    /// Samples a value from the truncated Gaussian distribution at `x` with the given
    /// `bandwidth` and domain `dom`.
    ///
    /// # Notes
    ///
    /// The sampling is performed using the inverse transform sampling method,
    /// which involves sampling a uniform random variable and applying the inverse CDF
    /// of the truncated Gaussian distribution.
    fn sample<R: Rng>(
        rng: &mut R,
        x: &f64,
        context: &Self::Context,
        _dom: &Unit,
    ) -> f64 {
        if (context.p_up - context.p_low).abs() < f64::EPSILON {
            return *x;
        }
        let u: f64 = rng.random_range(context.p_low..context.p_up);
        gaussian_icdf(*x, context.bandwidth, &u).clamp(0.0, 1.0)
    }
}

impl KernelFunc<Int> for GaussianKernel {
    type Context = GaussianContext;

    fn get_context(x: &i64, bandwidth: f64, dom: &Int) -> Self::Context {
        let lhs = 1. / (bandwidth * SQRT_2PI);
        let (low, up) = dom.get_bounds();
        let low = low as f64 - 0.5;
        let up = up as f64 + 0.5;

        let p_low = gaussian_cdf(x.as_(), bandwidth, &low);
        let p_up = gaussian_cdf(x.as_(), bandwidth, &up);
        let cst = gaussian_interval(x, bandwidth, low, up);
        GaussianContext { bandwidth, lhs, cst, p_low, p_up }
    }

    fn compute(
        x1: &i64,
        x2: &i64,
        context: &Self::Context,
        _dom: &Int,
    ) -> f64 {
        let x = *x1 as f64;
        let (low, up) = (x - 0.5, x + 0.5);
        let cdf = gaussian_interval(x2, context.bandwidth, low, up);
        cdf / context.cst
    }

    fn prior(x: &i64, dom: &Int) -> f64 {
        let (low, up) = dom.get_bounds();
        let low = low as f64 - 0.5;
        let up = up as f64 + 0.5;
        let mean = (low + up) / 2.0;
        let std = up - low;
        gaussian_pdf(mean, std, x.as_())
    }

    /// Samples a value from the truncated Gaussian distribution at `x` with the given
    /// `bandwidth` and domain `dom`.
    ///
    /// # Notes
    ///
    /// The sampling is performed using the inverse transform sampling method,
    /// which involves sampling a uniform random variable and applying the inverse CDF
    /// of the truncated Gaussian distribution.
    fn sample<R: Rng>(
        rng: &mut R,
        x: &i64,
        context: &Self::Context,
        dom: &Int,
    ) -> i64 {
        let (low, up) = dom.get_bounds();
        if (context.p_up - context.p_low).abs() < f64::EPSILON {
            return *x;
        }
        let u: f64 = rng.random_range(context.p_low..context.p_up);
        (gaussian_icdf(x.as_(), context.bandwidth, &u).round() as i64).clamp(low, up)
    }
}

impl KernelFunc<Nat> for GaussianKernel {
    type Context = GaussianContext;

    fn get_context(x: &u64, bandwidth: f64, dom: &Nat) -> Self::Context {

        let lhs = 1. / (bandwidth * SQRT_2PI);
        let (low, up) = dom.get_bounds();
        let low = low as f64 - 0.5;
        let up = up as f64 + 0.5;

        let p_low = gaussian_cdf(x.as_(), bandwidth, &low);
        let p_up = gaussian_cdf(x.as_(), bandwidth, &up);
        let cst = gaussian_interval(x, bandwidth, low, up);
        GaussianContext { bandwidth, lhs, cst, p_low, p_up }
    }

    fn compute(
        x1: &u64,
        x2: &u64,
        context: &Self::Context,
        _dom: &Nat,
    ) -> f64 {
        let x = *x1 as f64;
        let (low, up) = (x - 0.5, x + 0.5);
        let cdf = gaussian_interval(x2, context.bandwidth, low, up);
        cdf / context.cst
    }

    fn prior(x: &u64, dom: &Nat) -> f64 {
        let (low, up) = dom.get_bounds();
        let low = low as f64 - 0.5;
        let up = up as f64 + 0.5;
        let mean = (low + up) / 2.0;
        let std = up - low;
        gaussian_pdf(mean, std, x.as_())
    }

    /// Samples a value from the truncated Gaussian distribution at `x` with the given
    /// `bandwidth` and domain `dom`.
    ///
    /// # Notes
    ///
    /// The sampling is performed using the inverse transform sampling method,
    /// which involves sampling a uniform random variable and applying the inverse CDF
    /// of the truncated Gaussian distribution.
    fn sample<R: Rng>(
        rng: &mut R,
        x: &u64,
        context: &Self::Context,
        dom: &Nat,
    ) -> u64 {
        let (low, up) = dom.get_bounds();
        if (context.p_up - context.p_low).abs() < f64::EPSILON {
            return *x;
        }
        let u: f64 = rng.random_range(context.p_low..context.p_up);
        (gaussian_icdf(x.as_(), context.bandwidth, &u).round() as u64).clamp(low, up)
    }
}

/// The Aitchison-Aitken kernel for categorical domains $\mathcal{D}$ (e.g. [GridDom]) is defined as follows:
/// $$
/// K(x_1, x_2 \\,\\lvert\\, b) = \\begin{cases}
///     1 - b & \\text{if } x_1 = x_2 \\\\
///     \\frac{b}{|\mathcal{D}| - 1} & \\text{if } x_1 \\neq x_2
/// \\end{cases}\enspace\\\text{,}
/// $$
/// where $b$ is the bandwidth parameter.
pub struct AitchisonAitkenKernel;

#[derive(Serialize, Deserialize)]
pub struct AitchisonAitkenContext {
    pub bandwidth: f64,
}

impl AitchisonAitkenContext {
    pub fn new(bandwidth: f64) -> Self {
        AitchisonAitkenContext { bandwidth }
    }
}

impl<T: GridBounds> KernelFunc<GridDom<T>> for AitchisonAitkenKernel {
    type Context = AitchisonAitkenContext;

    fn get_context(
        _x: &<GridDom<T> as Domain>::TypeDom,
        bandwidth: f64,
        _dom: &GridDom<T>,
    ) -> Self::Context {
        AitchisonAitkenContext::new(bandwidth)
    }

    fn compute(
        x1: &T,
        x2: &T,
        context: &Self::Context,
        dom: &GridDom<T>,
    ) -> f64 {
        if x1 == x2 {
            1.0 - context.bandwidth
        } else {
            context.bandwidth / (dom.size() as f64 - 1.0)
        }
    }

    fn prior(_x: &T, dom: &GridDom<T>) -> f64 {
        1.0 / (dom.size() as f64)
    }

    fn sample<R: Rng>(
        rng: &mut R,
        x: &T,
        context: &Self::Context,
        dom: &GridDom<T>,
    ) -> T {
        let u: f64 = rng.random();
        let threshold = 1.0 - context.bandwidth;
        if u < threshold {
            x.clone()
        } else {
            // Sample a different category than x with equal probability
            let other_categories: Vec<&T> = dom.get_features().iter().filter(|&c| c != x).collect();
            let idx = rng.random_range(0..other_categories.len());
            other_categories[idx].clone()
        }
    }
}

impl KernelFunc<Bool> for AitchisonAitkenKernel {
    type Context = AitchisonAitkenContext;

    fn get_context(
        _x: &<Bool as Domain>::TypeDom,
        bandwidth: f64,
        _dom: &Bool,
    ) -> Self::Context {
        AitchisonAitkenContext::new(bandwidth)
    }

    fn compute(
        x1: &bool,
        x2: &bool,
        context: &Self::Context,
        dom: &Bool,
    ) -> f64 {
        if x1 == x2 {
            1.0 - context.bandwidth
        } else {
            context.bandwidth / (dom.size() as f64 - 1.0)
        }
    }

    fn prior(_x: &bool, dom: &Bool) -> f64 {
        1.0 / (dom.size() as f64)
    }

    fn sample<R: Rng>(
        rng: &mut R,
        x: &bool,
        context: &Self::Context,
        _dom: &Bool,
    ) -> bool {
        let u: f64 = rng.random();
        let threshold = 1.0 - context.bandwidth;
        if u < threshold { *x } else { !*x }
    }
}

pub enum MixedKernel {
    Gaussian(GaussianKernel),
    AitchisonAitken(AitchisonAitkenKernel),
}

#[derive(Serialize, Deserialize)]
pub enum MixedContext {
    Gaussian(GaussianContext),
    AitchisonAitken(AitchisonAitkenContext),
}

impl KernelFunc<Mixed> for MixedKernel {
    type Context = MixedContext;

    fn get_context(x: &MixedTypeDom, bandwidth: f64, dom: &Mixed) -> Self::Context {
        match (x, dom) {
            (MixedTypeDom::Real(x), Mixed::Real(d)) => {
                MixedContext::Gaussian(GaussianKernel::get_context(x, bandwidth, d))
            }
            (MixedTypeDom::Unit(x), Mixed::Unit(d)) => {
                MixedContext::Gaussian(GaussianKernel::get_context(x, bandwidth, d))
            }
            (MixedTypeDom::Int(x), Mixed::Int(d)) => {
                MixedContext::Gaussian(GaussianKernel::get_context(x, bandwidth, d))
            }
            (MixedTypeDom::Nat(x), Mixed::Nat(d)) => {
                MixedContext::Gaussian(GaussianKernel::get_context(x, bandwidth, d))
            }
            (MixedTypeDom::Bool(x), Mixed::Bool(d)) => {
                MixedContext::AitchisonAitken(AitchisonAitkenKernel::get_context(x, bandwidth, d))
            }
            (MixedTypeDom::Cat(x), Mixed::Cat(d)) => {
                MixedContext::AitchisonAitken(AitchisonAitkenKernel::get_context(x, bandwidth, d))
            }
            (MixedTypeDom::GridReal(x), Mixed::GridReal(d)) => {
                MixedContext::AitchisonAitken(AitchisonAitkenKernel::get_context(x, bandwidth, d))
            }
            (MixedTypeDom::GridNat(x), Mixed::GridNat(d)) => {
                MixedContext::AitchisonAitken(AitchisonAitkenKernel::get_context(x, bandwidth, d))
            }
            (MixedTypeDom::GridInt(x), Mixed::GridInt(d)) => {
                MixedContext::AitchisonAitken(AitchisonAitkenKernel::get_context(x, bandwidth, d))
            }
            _ => panic!("Mismatched kernel context and input type"),
        }
    }

    fn compute(
        x1: &MixedTypeDom,
        x2: &MixedTypeDom,
        context: &Self::Context,
        dom: &Mixed,
    ) -> f64 {
        match (x1, x2, context, dom) {
            (
                MixedTypeDom::Real(x),
                MixedTypeDom::Real(y),
                MixedContext::Gaussian(ctx),
                Mixed::Real(d),
            ) => GaussianKernel::compute(x, y, ctx, d),
            (
                MixedTypeDom::Unit(x),
                MixedTypeDom::Unit(y),
                MixedContext::Gaussian(ctx),
                Mixed::Unit(d),
            ) => GaussianKernel::compute(x, y, ctx, d),
            (
                MixedTypeDom::Int(x),
                MixedTypeDom::Int(y),
                MixedContext::Gaussian(ctx),
                Mixed::Int(d),
            ) => GaussianKernel::compute(x, y, ctx, d),
            (
                MixedTypeDom::Nat(x),
                MixedTypeDom::Nat(y),
                MixedContext::Gaussian(ctx),
                Mixed::Nat(d),
            ) => GaussianKernel::compute(x, y, ctx, d),
            (
                MixedTypeDom::Bool(x),
                MixedTypeDom::Bool(y),
                MixedContext::AitchisonAitken(ctx),
                Mixed::Bool(d),
            ) => AitchisonAitkenKernel::compute(x, y, ctx, d),
            (
                MixedTypeDom::Cat(x),
                MixedTypeDom::Cat(y),
                MixedContext::AitchisonAitken(ctx),
                Mixed::Cat(d),
            ) => AitchisonAitkenKernel::compute(x, y, ctx, d),
            (
                MixedTypeDom::GridReal(x),
                MixedTypeDom::GridReal(y),
                MixedContext::AitchisonAitken(ctx),
                Mixed::GridReal(d),
            ) => AitchisonAitkenKernel::compute(x, y, ctx, d),
            (
                MixedTypeDom::GridNat(x),
                MixedTypeDom::GridNat(y),
                MixedContext::AitchisonAitken(ctx),
                Mixed::GridNat(d),
            ) => AitchisonAitkenKernel::compute(x, y, ctx, d),
            (
                MixedTypeDom::GridInt(x),
                MixedTypeDom::GridInt(y),
                MixedContext::AitchisonAitken(ctx),
                Mixed::GridInt(d),
            ) => AitchisonAitkenKernel::compute(x, y, ctx, d),
            _ => panic!("Mismatched kernel context and input type"),
        }
    }

    fn prior(x: &MixedTypeDom, dom: &Mixed) -> f64 {
        match (x, dom) {
            (MixedTypeDom::Real(x), Mixed::Real(d)) => GaussianKernel::prior(x, d),
            (MixedTypeDom::Unit(x), Mixed::Unit(d)) => GaussianKernel::prior(x, d),
            (MixedTypeDom::Int(x), Mixed::Int(d)) => GaussianKernel::prior(x, d),
            (MixedTypeDom::Nat(x), Mixed::Nat(d)) => GaussianKernel::prior(x, d),
            (MixedTypeDom::Bool(x), Mixed::Bool(d)) => AitchisonAitkenKernel::prior(x, d),
            (MixedTypeDom::Cat(x), Mixed::Cat(d)) => AitchisonAitkenKernel::prior(x, d),
            (MixedTypeDom::GridReal(x), Mixed::GridReal(d)) => AitchisonAitkenKernel::prior(x, d),
            (MixedTypeDom::GridNat(x), Mixed::GridNat(d)) => AitchisonAitkenKernel::prior(x, d),
            (MixedTypeDom::GridInt(x), Mixed::GridInt(d)) => AitchisonAitkenKernel::prior(x, d),
            _ => panic!("Mismatched kernel context and input type"),
        }
    }

    fn sample<R: Rng>(
        rng: &mut R,
        x: &MixedTypeDom,
        context: &Self::Context,
        dom: &Mixed,
    ) -> MixedTypeDom {
        match (x, context, dom) {
            (
                MixedTypeDom::Real(x),
                MixedContext::Gaussian(ctx),
                Mixed::Real(d),
            ) => MixedTypeDom::Real(GaussianKernel::sample(rng, x, ctx, d)),
            (
                MixedTypeDom::Unit(x),
                MixedContext::Gaussian(ctx),
                Mixed::Unit(d),
            ) => MixedTypeDom::Unit(GaussianKernel::sample(rng, x, ctx, d)),
            (
                MixedTypeDom::Int(x),
                MixedContext::Gaussian(ctx),
                Mixed::Int(d),
            ) => MixedTypeDom::Int(GaussianKernel::sample(rng, x, ctx, d)),
            (
                MixedTypeDom::Nat(x),
                MixedContext::Gaussian(ctx),
                Mixed::Nat(d),
            ) => MixedTypeDom::Nat(GaussianKernel::sample(rng, x, ctx, d)),
            (
                MixedTypeDom::Bool(x),
                MixedContext::AitchisonAitken(ctx),
                Mixed::Bool(d),
            ) => MixedTypeDom::Bool(AitchisonAitkenKernel::sample(rng, x, ctx, d)),
            (
                MixedTypeDom::Cat(x),
                MixedContext::AitchisonAitken(ctx),
                Mixed::Cat(d),
            ) => MixedTypeDom::Cat(AitchisonAitkenKernel::sample(rng, x, ctx, d)),
            (
                MixedTypeDom::GridReal(x),
                MixedContext::AitchisonAitken(ctx),
                Mixed::GridReal(d),
            ) => MixedTypeDom::GridReal(AitchisonAitkenKernel::sample(rng, x, ctx, d)),
            (
                MixedTypeDom::GridNat(x),
                MixedContext::AitchisonAitken(ctx),
                Mixed::GridNat(d),
            ) => MixedTypeDom::GridNat(AitchisonAitkenKernel::sample(rng, x, ctx, d)),
            (
                MixedTypeDom::GridInt(x),
                MixedContext::AitchisonAitken(ctx),
                Mixed::GridInt(d),
            ) => MixedTypeDom::GridInt(AitchisonAitkenKernel::sample(rng, x, ctx, d)),
            _ => panic!("Mismatched kernel context and input type"),
        }
    }
}

/// Trait for kernel functions used in the TPE algorithm.
/// The kernel function computes the similarity between a given solution and the points in the archive, which is used to estimate the density of good and bad solutions.
///
/// # See also
/// - [`Univariate`] for the univariate kernel, which assumes independence between dimensions and computes the product of the kernel values for each dimension.
/// - [`Multivariate`] for the multivariate kernel, which models the joint distribution of all dimensions.
pub trait Kernel<Dom, Scp, S, SolId, SInfo, Out>
where
    Self: Sized + Serialize + for<'a> Deserialize<'a>,
    Dom: Domain,
    Scp: Searchspace<S, SolId, SInfo, Opt = Dom>,
    S: Uncomputed<SolId, Dom, SInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    /// The context type for the kernel,
    /// which can hold precomputed values or parameters needed
    /// for efficient kernel computation for each point within the archive.
    type Context: Serialize + for<'a> Deserialize<'a>;

    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>;

    fn compute(
        &self,
        s: &S::Raw,
        archive: &[&Xy<S::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64;

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64;

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<S::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> S::Raw;
}

/// The univariate kernel computes the product of the kernel values for each dimension of the solution, assuming independence between dimensions.
/// For a solution of dimension $D$, the kernel value is computed as:
/// $$
/// K(\\mathbb{s}, \\{\\mathbb{s}\\}_{n=1}^N) = \\prod_{d=1}^{D} \\sum_{n=1}^N w_n K_d(\mathbb{s}_{d}, \\mathbb{s}_{n,d} \\,\\lvert\\, b_d)\\enspace\\text{,}
/// $$
/// where $K_d$ is the kernel function ([`KernelFunc`]) for the $d$-th dimension, and $b_d$ is the bandwidth parameter for that dimension.
/// The bandwidth is computed using the Optuna rule [`optuna_bw`] for numerical dimensions, and [`cat_bw`] for categorical dimensions.
#[derive(Serialize, Deserialize)]
pub struct Univariate;

impl<Scp, S, SolId, SInfo, Out> Kernel<Mixed, Scp, S, SolId, SInfo, Out> for Univariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Mixed> + HasVariables,
    S: Uncomputed<SolId, Mixed, SInfo, Raw = Arc<[TypeDom<Mixed>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type Context = Vec<MixedContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64 {
        let product: f64 = s
            .iter()
            .zip(scp.iter_opt())
            .enumerate()
            .map(|(d, (x1, dom))| {
                archive
                    .iter()
                    .zip(weights.weights.iter())
                    .zip(context.iter())
                    .map(|((comp, weight), ctx)| {
                        MixedKernel::compute(x1, &comp.ref_x()[d], &ctx[d], dom)
                            * weight
                    })
                    .sum::<f64>()
            })
            .product();
        let prior = <Univariate as Kernel<Mixed, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + product
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> <S>::Raw {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &context[rng_idx];
                MixedKernel::sample(rng, &x[d], &ctx[d], dom)
            })
            .collect()
    }

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64 {
        s.iter()
            .enumerate()
            .map(|(d, x)| {
                let dom = scp.opt_at(d).unwrap();
                MixedKernel::prior(x, dom)
            })
            .product()
    }

    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>
    {
        archive
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                p.as_ref().ref_x()
                    .iter()
                    .zip(scp.iter_opt())
                    .enumerate()
                    .map(
                        |(dim, (x, dom))| 
                        MixedKernel::get_context(x, bw.get(idx, dim), dom)
                    )
                    .collect()
            })
            .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Kernel<Real, Scp, S, SolId, SInfo, Out> for Univariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Real> + HasVariables,
    S: Uncomputed<SolId, Real, SInfo, Raw = Arc<[TypeDom<Real>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type Context = Vec<GaussianContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64 {
        let product: f64 = s
            .iter()
            .zip(scp.iter_opt())
            .enumerate()
            .map(|(d, (x1, dom))| {
                archive
                    .iter()
                    .zip(weights.weights.iter())
                    .zip(context.iter())
                    .map(|((comp, weight), ctx)| {
                        GaussianKernel::compute(x1, &comp.ref_x()[d], &ctx[d], dom)
                            * weight
                    })
                    .sum::<f64>()
            })
            .product();
        let prior = <Univariate as Kernel<Real, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + product
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> <S>::Raw {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &context[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], dom)
            })
            .collect()
    }

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64 {
        s.iter()
            .enumerate()
            .map(|(d, x)| {
                let dom = scp.opt_at(d).unwrap();
                GaussianKernel::prior(x, dom)
            })
            .product()
    }
    
    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>
    {
        archive
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                p.as_ref().ref_x()
                    .iter()
                    .zip(scp.iter_opt())
                    .enumerate()
                    .map(
                        |(dim, (x, dom))| 
                        GaussianKernel::get_context(x, bw.get(idx, dim), dom)
                    )
                    .collect()
            })
            .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Kernel<Int, Scp, S, SolId, SInfo, Out> for Univariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Int> + HasVariables,
    S: Uncomputed<SolId, Int, SInfo, Raw = Arc<[TypeDom<Int>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type Context = Vec<GaussianContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64 {
        let product: f64 = s
            .iter()
            .zip(scp.iter_opt())
            .enumerate()
            .map(|(d, (x1, dom))| {
                archive
                    .iter()
                    .zip(weights.weights.iter())
                    .zip(context.iter())
                    .map(|((comp, weight), ctx)| {
                        GaussianKernel::compute(x1, &comp.ref_x()[d], &ctx[d], dom)
                            * weight
                    })
                    .sum::<f64>()
            })
            .product();
        let prior = <Univariate as Kernel<Int, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + product
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> <S>::Raw {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &context[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], dom)
            })
            .collect()
    }

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64 {
        s.iter()
            .enumerate()
            .map(|(d, x)| {
                let dom = scp.opt_at(d).unwrap();
                GaussianKernel::prior(x, dom)
            })
            .product()
    }

    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>
    {
        archive
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                p.as_ref().ref_x()
                    .iter()
                    .zip(scp.iter_opt())
                    .enumerate()
                    .map(
                        |(dim, (x, dom))| 
                        GaussianKernel::get_context(x, bw.get(idx, dim), dom)
                    )
                    .collect()
            })
            .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Kernel<Nat, Scp, S, SolId, SInfo, Out> for Univariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Nat> + HasVariables,
    S: Uncomputed<SolId, Nat, SInfo, Raw = Arc<[TypeDom<Nat>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type Context = Vec<GaussianContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64 {
        let product: f64 = s
            .iter()
            .zip(scp.iter_opt())
            .enumerate()
            .map(|(d, (x1, dom))| {
                archive
                    .iter()
                    .zip(weights.weights.iter())
                    .zip(context.iter())
                    .map(|((comp, weight), ctx)| {
                        GaussianKernel::compute(x1, &comp.ref_x()[d], &ctx[d], dom)
                            * weight
                    })
                    .sum::<f64>()
            })
            .product();
        let prior = <Univariate as Kernel<Nat, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + product
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> <S>::Raw {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &context[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], dom)
            })
            .collect()
    }

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64 {
        s.iter()
            .enumerate()
            .map(|(d, x)| {
                let dom = scp.opt_at(d).unwrap();
                GaussianKernel::prior(x, dom)
            })
            .product()
    }

    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>
    {
        archive
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                p.as_ref().ref_x()
                    .iter()
                    .zip(scp.iter_opt())
                    .enumerate()
                    .map(
                        |(dim, (x, dom))| 
                        GaussianKernel::get_context(x, bw.get(idx, dim), dom)
                    )
                    .collect()
            })
            .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Kernel<Unit, Scp, S, SolId, SInfo, Out> for Univariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Unit> + HasVariables,
    S: Uncomputed<SolId, Unit, SInfo, Raw = Arc<[TypeDom<Unit>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type Context = Vec<GaussianContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64 {
        let product: f64 = s
            .iter()
            .zip(scp.iter_opt())
            .enumerate()
            .map(|(d, (x1, dom))| {
                archive
                    .iter()
                    .zip(weights.weights.iter())
                    .zip(context.iter())
                    .map(|((comp, weight), ctx)| {
                        GaussianKernel::compute(x1, &comp.ref_x()[d], &ctx[d], dom)
                            * weight
                    })
                    .sum::<f64>()
            })
            .product();
        let prior = <Univariate as Kernel<Unit, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + product
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> <S>::Raw {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &context[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], dom)
            })
            .collect()
    }

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64 {
        s.iter()
            .enumerate()
            .map(|(d, x)| {
                let dom = scp.opt_at(d).unwrap();
                GaussianKernel::prior(x, dom)
            })
            .product()
    }

    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>
    {
        archive
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                p.as_ref().ref_x()
                    .iter()
                    .zip(scp.iter_opt())
                    .enumerate()
                    .map(
                        |(dim, (x, dom))| 
                        GaussianKernel::get_context(x, bw.get(idx, dim), dom)
                    )
                    .collect()
            })
            .collect()
    }
}

impl<G, Scp, S, SolId, SInfo, Out> Kernel<GridDom<G>, Scp, S, SolId, SInfo, Out> for Univariate
where
    G: GridBounds,
    Scp: Searchspace<S, SolId, SInfo, Opt = GridDom<G>> + HasVariables,
    S: Uncomputed<SolId, GridDom<G>, SInfo, Raw = Arc<[TypeDom<GridDom<G>>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type Context = Vec<AitchisonAitkenContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64 {
        let product: f64 = s
            .iter()
            .zip(scp.iter_opt())
            .enumerate()
            .map(|(d, (x1, dom))| {
                archive
                    .iter()
                    .zip(weights.weights.iter())
                    .zip(context.iter())
                    .map(|((comp, weight), ctx)| {
                        AitchisonAitkenKernel::compute(x1, &comp.ref_x()[d], &ctx[d], dom)
                            * weight
                    })
                    .sum::<f64>()
            })
            .product();
        let prior = <Univariate as Kernel<GridDom<G>, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + product
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> <S>::Raw {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &context[rng_idx];
                AitchisonAitkenKernel::sample(rng, &x[d], &ctx[d], dom)
            })
            .collect()
    }

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64 {
        s.iter()
            .enumerate()
            .map(|(d, x)| {
                let dom = scp.opt_at(d).unwrap();
                AitchisonAitkenKernel::prior(x, dom)
            })
            .product()
    }

    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>
    {
        archive
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                p.as_ref().ref_x()
                    .iter()
                    .zip(scp.iter_opt())
                    .enumerate()
                    .map(
                        |(dim, (x, dom))| 
                        AitchisonAitkenKernel::get_context(x, bw.get(idx, dim), dom)
                    )
                    .collect()
            })
            .collect()
    }
}

/// The multivariate kernel computes the kernel value between two solutions by considering the joint distribution of all dimensions, without assuming independence.
/// The kernel value is computed as:
/// For a solution of dimension $D$, the kernel value is computed as:
/// $$
/// K(\\mathbb{s}, \\{\\mathbb{s}\\}_{n=1}^N) = \\sum_{n=1}^N w_n \\prod_{d=1}^{D} K_d(\mathbb{s}_{d}, \\mathbb{s}_{n,d} \\,\\lvert\\, b_d)\\enspace\\text{,}
/// $$
/// where $K_d$ is the kernel function ([`KernelFunc`]) for the $d$-th dimension, and $b_d$ is the bandwidth parameter for that dimension.
/// The bandwidth is computed using the Optuna rule [`optuna_bw`] for numerical dimensions, and [`cat_bw`] for categorical dimensions.
#[derive(Serialize, Deserialize)]
pub struct Multivariate;

impl<Scp, S, SolId, SInfo, Out> Kernel<Mixed, Scp, S, SolId, SInfo, Out> for Multivariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Mixed> + HasVariables,
    S: Uncomputed<SolId, Mixed, SInfo, Raw = Arc<[TypeDom<Mixed>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type Context = Vec<MixedContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64 {
        let sum: f64 = archive
            .iter()
            .zip(weights.weights.iter())
            .zip(context.iter())
            .map(|((comp, weight), ctx)| {
                s.iter()
                    .zip(comp.ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), dctx)| {
                        MixedKernel::compute(x1, x2, dctx, dom)
                    })
                    .product::<f64>()
                    * weight
            })
            .sum();
        let prior = <Multivariate as Kernel<Mixed, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> <S>::Raw {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &context[rng_idx];
                MixedKernel::sample(rng, &x[d], &ctx[d], dom)
            })
            .collect()
    }

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64 {
        s.iter()
            .enumerate()
            .map(|(d, x)| {
                let dom = scp.opt_at(d).unwrap();
                MixedKernel::prior(x, dom)
            })
            .product()
    }

    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>
    {
        archive
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                p.as_ref().ref_x()
                    .iter()
                    .zip(scp.iter_opt())
                    .enumerate()
                    .map(
                        |(dim, (x, dom))| 
                        MixedKernel::get_context(x, bw.get(idx, dim), dom)
                    )
                    .collect()
            })
            .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Kernel<Real, Scp, S, SolId, SInfo, Out> for Multivariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Real> + HasVariables,
    S: Uncomputed<SolId, Real, SInfo, Raw = Arc<[TypeDom<Real>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type Context = Vec<GaussianContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64 {
        let sum: f64 = archive
            .iter()
            .zip(weights.weights.iter())
            .zip(context.iter())
            .map(|((comp, weight), ctx)| {
                s.iter()
                    .zip(comp.ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), dctx)| {
                        GaussianKernel::compute(x1, x2, dctx, dom)
                    })
                    .product::<f64>()
                    * weight
            })
            .sum();
        let prior = <Multivariate as Kernel<Real, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> <S>::Raw {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &context[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], dom)
            })
            .collect()
    }

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64 {
        s.iter()
            .enumerate()
            .map(|(d, x)| {
                let dom = scp.opt_at(d).unwrap();
                GaussianKernel::prior(x, dom)
            })
            .product()
    }

    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>
    {
        archive
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                p.as_ref().ref_x()
                    .iter()
                    .zip(scp.iter_opt())
                    .enumerate()
                    .map(
                        |(dim, (x, dom))| 
                        GaussianKernel::get_context(x, bw.get(idx, dim), dom)
                    )
                    .collect()
            })
            .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Kernel<Int, Scp, S, SolId, SInfo, Out> for Multivariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Int> + HasVariables,
    S: Uncomputed<SolId, Int, SInfo, Raw = Arc<[TypeDom<Int>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type Context = Vec<GaussianContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64 {
        let sum: f64 = archive
            .iter()
            .zip(weights.weights.iter())
            .zip(context.iter())
            .map(|((comp, weight), ctx)| {
                s.iter()
                    .zip(comp.ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), dctx)| {
                        GaussianKernel::compute(x1, x2, dctx, dom)
                    })
                    .product::<f64>()
                    * weight
            })
            .sum();
        let prior = <Multivariate as Kernel<Int, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> <S>::Raw {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &context[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], dom)
            })
            .collect()
    }

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64 {
        s.iter()
            .enumerate()
            .map(|(d, x)| {
                let dom = scp.opt_at(d).unwrap();
                GaussianKernel::prior(x, dom)
            })
            .product()
    }

    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>
    {
        archive
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                p.as_ref().ref_x()
                    .iter()
                    .zip(scp.iter_opt())
                    .enumerate()
                    .map(
                        |(dim, (x, dom))| 
                        GaussianKernel::get_context(x, bw.get(idx, dim), dom)
                    )
                    .collect()
            })
            .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Kernel<Nat, Scp, S, SolId, SInfo, Out> for Multivariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Nat> + HasVariables,
    S: Uncomputed<SolId, Nat, SInfo, Raw = Arc<[TypeDom<Nat>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type Context = Vec<GaussianContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64 {
        let sum: f64 = archive
            .iter()
            .zip(weights.weights.iter())
            .zip(context.iter())
            .map(|((comp, weight), ctx)| {
                s.iter()
                    .zip(comp.ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), dctx)| {
                        GaussianKernel::compute(x1, x2, dctx, dom)
                    })
                    .product::<f64>()
                    * weight
            })
            .sum();
        let prior = <Multivariate as Kernel<Nat, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> <S>::Raw {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &context[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], dom)
            })
            .collect()
    }

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64 {
        s.iter()
            .enumerate()
            .map(|(d, x)| {
                let dom = scp.opt_at(d).unwrap();
                GaussianKernel::prior(x, dom)
            })
            .product()
    }

    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>
    {
        archive
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                p.as_ref().ref_x()
                    .iter()
                    .zip(scp.iter_opt())
                    .enumerate()
                    .map(
                        |(dim, (x, dom))| 
                        GaussianKernel::get_context(x, bw.get(idx, dim), dom)
                    )
                    .collect()
            })
            .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Kernel<Unit, Scp, S, SolId, SInfo, Out> for Multivariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Unit> + HasVariables,
    S: Uncomputed<SolId, Unit, SInfo, Raw = Arc<[TypeDom<Unit>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type Context = Vec<GaussianContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64 {
        let sum: f64 = archive
            .iter()
            .zip(weights.weights.iter())
            .zip(context.iter())
            .map(|((comp, weight), ctx)| {
                s.iter()
                    .zip(comp.ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), dctx)| {
                        GaussianKernel::compute(x1, x2, dctx, dom)
                    })
                    .product::<f64>()
                    * weight
            })
            .sum();
        let prior = <Multivariate as Kernel<Unit, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> <S>::Raw {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &context[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], dom)
            })
            .collect()
    }

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64 {
        s.iter()
            .enumerate()
            .map(|(d, x)| {
                let dom = scp.opt_at(d).unwrap();
                GaussianKernel::prior(x, dom)
            })
            .product()
    }

    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>
    {
        archive
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                p.as_ref().ref_x()
                    .iter()
                    .zip(scp.iter_opt())
                    .enumerate()
                    .map(
                        |(dim, (x, dom))| 
                        GaussianKernel::get_context(x, bw.get(idx, dim), dom)
                    )
                    .collect()
            })
            .collect()
    }
}

impl<G, Scp, S, SolId, SInfo, Out> Kernel<GridDom<G>, Scp, S, SolId, SInfo, Out> for Multivariate
where
    G: GridBounds,
    Scp: Searchspace<S, SolId, SInfo, Opt = GridDom<G>> + HasVariables,
    S: Uncomputed<SolId, GridDom<G>, SInfo, Raw = Arc<[TypeDom<GridDom<G>>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type Context = Vec<AitchisonAitkenContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64 {
        let sum: f64 = archive
            .iter()
            .zip(weights.weights.iter())
            .zip(context.iter())
            .map(|((comp, weight), ctx)| {
                s.iter()
                    .zip(comp.ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), dctx)| {
                        AitchisonAitkenKernel::compute(x1, x2, dctx, dom)
                    })
                    .product::<f64>()
                    * weight
            })
            .sum();
        let prior = <Multivariate as Kernel<GridDom<G>, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> <S>::Raw {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &context[rng_idx];
                AitchisonAitkenKernel::sample(rng, &x[d], &ctx[d], dom)
            })
            .collect()
    }

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64 {
        s.iter()
            .enumerate()
            .map(|(d, x)| {
                let dom = scp.opt_at(d).unwrap();
                AitchisonAitkenKernel::prior(x, dom)
            })
            .product()
    }

    fn get_context<T: AsRef<Xy<S::Raw, TypeCodom<Out>>>, Bw: BandwidthType>(archive: &[T], scp: &Scp, bw: &Bw) -> Vec<Self::Context>
    {
        archive
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                p.as_ref().ref_x()
                    .iter()
                    .zip(scp.iter_opt())
                    .enumerate()
                    .map(
                        |(dim, (x, dom))| 
                        AitchisonAitkenKernel::get_context(x, bw.get(idx, dim), dom)
                    )
                    .collect()
            })
            .collect()
    }
}
