use tantale_core::{
    Bool, Domain, GridDom, HasX, Id, Int, Mixed, MixedTypeDom, Nat, Outcome, Real, Searchspace, SolInfo, Uncomputed, Unit, Xy, domain::{CategoricalDomain, NumericalDomain, TypeDom, codomain::TypeCodom, grid::GridBounds}, has_trait::HasVariables
};

use num::{Num, cast::AsPrimitive};
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use statrs::function::erf;
use core::f64;
use std::sync::Arc;

use crate::{bayesian::{
    bandwidth::{cat_bw, optuna_bw},
    weighter::PointWeights,
}};

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

pub fn gaussian_cdf<T:  Num + AsPrimitive<f64>>(mean: f64, std: f64, x: &T) -> f64 {
    0.5 * erf::erfc((mean - x.as_()) / (std * f64::consts::SQRT_2))
}

pub fn gaussian_icdf<T:  Num + AsPrimitive<f64>>(mean: f64, std: f64, x: &T) -> f64 {
    mean - (std * f64::consts::SQRT_2 * erf::erfc_inv(2.0 * x.as_()))
}

/// Computes :
/// $$
/// \\begin{split}
///     \\mathbb{P}\\left( L < X = x_2 < U \\right) &= \\int_{L}^{U} K(x,x_2\\,|\\,b)dx \\\\
///                 &= \\frac{1}{2}\\left(\\text{erf}(\\frac{(R - x_2)}{(\\sqrt{2} b)}) -\\text{erf}(\\frac{(L - x_2)}{(\\sqrt{2} b)}) \\right)
/// \\end{split}
/// $$
fn gaussian_interval<T:  Num + AsPrimitive<f64>>(x: &T, bandwidth: f64, low: f64, up: f64) -> f64
{
    let x_f = x.as_();
    let denom = f64::consts::SQRT_2 * bandwidth;
    let low_erf = erf::erf((low - x_f) / denom);
    let up_erf = erf::erf((up - x_f) / denom);
    (up_erf - low_erf) / 2.0
}

pub trait KernelFunc<Dom: Domain>
{
    /// The shared context type for the kernel,
    /// which can hold precomputed values or parameters needed for efficient kernel computation across multiple calls. 
    /// This is useful for kernels that require expensive computations that can be reused, such as normalization constants or bandwidth parameters.
    type SContext: Serialize + for<'a> Deserialize<'a>;
    /// The context type for the kernel, 
    /// which can hold precomputed values or parameters needed 
    /// for efficient kernel computation.
    type KContext: Serialize + for<'a> Deserialize<'a>;

    fn default_scontext(dom: &Dom) -> Self::SContext;
    fn update_scontext(bandwidth: f64, scontext: &mut Self::SContext, dom: &Dom);
    
    fn default_kcontext(dom: &Dom) -> Self::KContext;
    fn update_kcontext(x: &Dom::TypeDom, kcontext: &mut Self::KContext, scontext: &Self::SContext, dom: &Dom);

    /// Computes the kernel function between two [`SolutionShape`] instances of `Opt` type `Dom`.
    ///
    /// # Arguments
    /// * `x1` - The first point.
    /// * `x2` - The second point.
    /// * `bandwidth` - The bandwidth parameter of the kernel.
    /// * `dom` - The domain of the solutions.
    ///
    /// # Returns
    /// The kernel value between the two [`SolutionShape`].
    fn compute(x1: &Dom::TypeDom, x2: &Dom::TypeDom, kcontext: &Self::KContext, scontext: &Self::SContext, dom: &Dom) -> f64;

    /// Computes the prior probability for a given point `x` in the domain `dom`.
    fn prior(x: &Dom::TypeDom, dom: &Dom) -> f64;

    /// Samples a value from the kernel distribution at `x` with the given `bandwidth` and domain `dom`.
    fn sample<R: Rng>(
        rng: &mut R,
        x: &Dom::TypeDom,
        context: &Self::KContext,
        scontext: &Self::SContext,
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
pub struct GaussianSContext {
    pub bandwidth: f64,
    pub lhs: f64,
}

impl GaussianSContext {
    pub fn new(bandwidth: f64, lhs: f64) -> Self
    {
        GaussianSContext {
            bandwidth,
            lhs,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct GaussianKContext {
    pub cst: f64,
    pub p_low: f64,
    pub p_up: f64,
}

impl KernelFunc<Real> for GaussianKernel {
    type SContext = GaussianSContext;
    type KContext = GaussianKContext;
    
    fn default_scontext(_dom: &Real) -> Self::SContext {
        GaussianSContext::new(1.0, 0.0)
    }
    fn update_scontext(bandwidth: f64, scontext: &mut Self::SContext, _dom: &Real)
    {
        scontext.lhs = 1. / (bandwidth * SQRT_2PI);
        scontext.bandwidth = bandwidth;
    }
    
    fn default_kcontext(_dom: &Real) -> Self::KContext {
        GaussianKContext { cst : 0.0, p_low: 0.0, p_up: 0.0 }
    }

    fn update_kcontext(x: &f64, kcontext: &mut Self::KContext, scontext: &Self::SContext, dom: &Real) {
        let (low, up) = dom.get_bounds();
        let cst = gaussian_interval(x, scontext.bandwidth, low, up);
        let p_low = gaussian_cdf(*x, scontext.bandwidth, &low);
        let p_up = gaussian_cdf(*x, scontext.bandwidth, &up);
        kcontext.cst = cst;
        kcontext.p_low = p_low;
        kcontext.p_up = p_up;
    }

    fn compute(x1: &f64, x2: &f64, kcontext: &Self::KContext, scontext: &Self::SContext, _dom: &Real) -> f64 {
        scontext.lhs * (-0.5 * ((x1 - x2) / scontext.bandwidth).powi(2)).exp() / kcontext.cst
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
        context: &Self::KContext,
        scontext: &Self::SContext,
        dom: &Real,
    ) -> f64
    {
        let (low, up) = dom.get_bounds();
        if (context.p_up - context.p_low).abs() < f64::EPSILON {
            return *x;
        }
        let u: f64 = rng.random_range(context.p_low..context.p_up);
        gaussian_icdf(*x, scontext.bandwidth, &u).clamp(low, up)
    }
}

impl KernelFunc<Unit> for GaussianKernel
{
    type SContext = GaussianSContext;
    type KContext = GaussianKContext;
    
    fn default_scontext(_dom: &Unit) -> Self::SContext {
        GaussianSContext::new(1.0, 0.0)
    }
    fn update_scontext(bandwidth: f64, scontext: &mut Self::SContext, _dom: &Unit) {
        scontext.lhs = 1. / (bandwidth * SQRT_2PI);
        scontext.bandwidth = bandwidth;
    }
    
    fn default_kcontext(_dom: &Unit) -> Self::KContext {
        GaussianKContext { cst : 0.0, p_low: 0.0, p_up: 0.0 }
    }

    fn update_kcontext(x: &f64, kcontext: &mut Self::KContext, scontext: &Self::SContext, _dom: &Unit) {
        let cst = gaussian_interval(x, scontext.bandwidth, 0.0, 1.0);
        let p_low = gaussian_cdf(*x, scontext.bandwidth, &0.0);
        let p_up = gaussian_cdf(*x, scontext.bandwidth, &1.0);
        kcontext.cst = cst;
        kcontext.p_low = p_low;
        kcontext.p_up = p_up;
    }

    fn compute(x1: &f64, x2: &f64, kcontext: &Self::KContext, scontext: &Self::SContext, _dom: &Unit) -> f64 {
        scontext.lhs * (-0.5 * ((x1 - x2) / scontext.bandwidth).powi(2)).exp() / kcontext.cst
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
        context: &Self::KContext,
        scontext: &Self::SContext,
        _dom: &Unit,
    ) -> f64
    {
        if (context.p_up - context.p_low).abs() < f64::EPSILON {
            return *x;
        }
        let u: f64 = rng.random_range(context.p_low..context.p_up);
        gaussian_icdf(*x, scontext.bandwidth, &u).clamp(0.0, 1.0)
    }
}

impl KernelFunc<Int> for GaussianKernel 
{
    type SContext = GaussianSContext;
    type KContext = GaussianKContext;
    
    fn default_scontext(_dom: &Int) -> Self::SContext {
        GaussianSContext::new(1.0, 0.0)
    }
    fn update_scontext(bandwidth: f64, scontext: &mut Self::SContext, _dom: &Int) {
        scontext.lhs = 1. / (bandwidth * SQRT_2PI);
        scontext.bandwidth = bandwidth;
    }
    
    fn default_kcontext(_dom: &Int) -> Self::KContext {
        GaussianKContext { cst : 0.0, p_low: 0.0, p_up: 0.0 }
    }

    fn update_kcontext(x: &i64, kcontext: &mut Self::KContext, scontext: &Self::SContext, dom: &Int) {
        let (low, up) = dom.get_bounds();
        let low = low as f64 - 0.5;
        let up = up as f64 + 0.5;

        let p_low = gaussian_cdf(x.as_(), scontext.bandwidth, &low);
        let p_up = gaussian_cdf(x.as_(), scontext.bandwidth, &up);
        let cst = gaussian_interval(x, scontext.bandwidth, low, up);
        kcontext.cst = cst;
        kcontext.p_low = p_low;
        kcontext.p_up = p_up;
    }

    fn compute(x1: &i64, x2: &i64, kcontext: &Self::KContext, scontext: &Self::SContext, _dom: &Int) -> f64 {
        let x= *x1 as f64;
        let (low, up) = (x - 0.5, x + 0.5);
        let cdf = gaussian_interval(x2, scontext.bandwidth, low, up);
        cdf / kcontext.cst
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
        context: &Self::KContext,
        scontext: &Self::SContext,
        dom: &Int,
    ) -> i64
    {
        let (low, up) = dom.get_bounds();
        if (context.p_up - context.p_low).abs() < f64::EPSILON {
            return *x;
        }
        let u: f64 = rng.random_range(context.p_low..context.p_up);
        (gaussian_icdf(x.as_(), scontext.bandwidth, &u).round() as i64).clamp(low, up)
    }
}

impl KernelFunc<Nat> for GaussianKernel
{
    type SContext = GaussianSContext;
    type KContext = GaussianKContext;
    
    fn default_scontext(_dom: &Nat) -> Self::SContext {
        GaussianSContext::new(1.0, 0.0)
    }
    fn update_scontext(bandwidth: f64, scontext: &mut Self::SContext, _dom: &Nat) {
        scontext.lhs = 1. / (bandwidth * SQRT_2PI);
        scontext.bandwidth = bandwidth;
    }
    
    fn default_kcontext(_dom: &Nat) -> Self::KContext {
        GaussianKContext { cst : 0.0, p_low: 0.0, p_up: 0.0 }
    }

    fn update_kcontext(x: &u64, kcontext: &mut Self::KContext, scontext: &Self::SContext, dom: &Nat) {
        let (low, up) = dom.get_bounds();
        let low = low as f64 - 0.5;
        let up = up as f64 + 0.5;

        let p_low = gaussian_cdf(x.as_(), scontext.bandwidth, &low);
        let p_up = gaussian_cdf(x.as_(), scontext.bandwidth, &up);
        let cst = gaussian_interval(x, scontext.bandwidth, low, up);
        kcontext.cst = cst;
        kcontext.p_low = p_low;
        kcontext.p_up = p_up;
    }

    fn compute(x1: &u64, x2: &u64, kcontext: &Self::KContext, scontext: &Self::SContext, _dom: &Nat) -> f64 {
        let x= *x1 as f64;
        let (low, up) = (x - 0.5, x + 0.5);
        let cdf = gaussian_interval(x2, scontext.bandwidth, low, up);
        cdf / kcontext.cst
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
        context: &Self::KContext,
        scontext: &Self::SContext,
        dom: &Nat,
    ) -> u64
    {
        let (low, up) = dom.get_bounds();
        if (context.p_up - context.p_low).abs() < f64::EPSILON {
            return *x;
        }
        let u: f64 = rng.random_range(context.p_low..context.p_up);
        (gaussian_icdf(x.as_(), scontext.bandwidth, &u).round() as u64).clamp(low, up)
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
pub struct AitchisonAitkenSContext{
    pub bandwidth: f64, 
}

impl AitchisonAitkenSContext {
    pub fn new(bandwidth: f64) -> Self {
        AitchisonAitkenSContext {
            bandwidth,
        }
    }
}

impl<T: GridBounds> KernelFunc<GridDom<T>> for AitchisonAitkenKernel
{

    type SContext = AitchisonAitkenSContext;
    type KContext = (); // No context needed for categorical kernel
    
    fn default_scontext(_dom: &GridDom<T>) -> Self::SContext {
        AitchisonAitkenSContext {
            bandwidth: 1.0, // Placeholder, should be set based on the data
        }
    }

    fn update_scontext(bandwidth: f64, scontext: &mut Self::SContext, _dom: &GridDom<T>) {
        scontext.bandwidth = bandwidth;
    }

    fn default_kcontext(_dom: &GridDom<T>) -> Self::KContext { }

    fn update_kcontext(_x: &T, _kcontext: &mut Self::KContext, _scontext: &Self::SContext, _dom: &GridDom<T>) { }

    fn compute(x1: &T, x2: &T, _kcontext: &Self::KContext, scontext: &Self::SContext, dom: &GridDom<T>) -> f64 {
        if x1 == x2 {
            1.0 - scontext.bandwidth
        } else {
            scontext.bandwidth / (dom.size() as f64 - 1.0)
        }
    }

    fn prior(_x: &T, dom: &GridDom<T>) -> f64 {
        1.0 / (dom.size() as f64)
    }

    fn sample<R: Rng>(
        rng: &mut R,
        x: &T,
        _context: &Self::KContext,
        scontext: &Self::SContext,
        dom: &GridDom<T>,
    ) -> T
    {
        let u: f64 = rng.random();
        let threshold = 1.0 - scontext.bandwidth;
        if u < threshold {
            x.clone()
        } else {
            // Sample a different category than x with equal probability
            let other_categories: Vec<&T> =
                dom.get_features().iter().filter(|&c| c != x).collect();
            let idx = rng.random_range(0..other_categories.len());
            other_categories[idx].clone()
        }
    }
}

impl KernelFunc<Bool> for AitchisonAitkenKernel
{
    type SContext = AitchisonAitkenSContext;
    type KContext = (); // No context needed for categorical kernel
    
    fn default_scontext(_dom: &Bool) -> Self::SContext {
        AitchisonAitkenSContext {
            bandwidth: 1.0, // Placeholder, should be set based on the data
        }
    }

    fn update_scontext(bandwidth: f64, scontext: &mut Self::SContext, _dom: &Bool) {
        scontext.bandwidth = bandwidth;
    }

    fn default_kcontext(_dom: &Bool) -> Self::KContext { }

    fn update_kcontext(_x: &bool, _kcontext: &mut Self::KContext, _scontext: &Self::SContext, _dom: &Bool) { }

    fn compute(x1: &bool, x2: &bool, _kcontext: &Self::KContext, scontext: &Self::SContext, dom: &Bool) -> f64 {
        if x1 == x2 {
            1.0 - scontext.bandwidth
        } else {
            scontext.bandwidth / (dom.size() as f64 - 1.0)
        }
    }

    fn prior(_x: &bool, dom: &Bool) -> f64 {
        1.0 / (dom.size() as f64)
    }

    fn sample<R: Rng>(
        rng: &mut R,
        x: &bool,
        _context: &Self::KContext,
        scontext: &Self::SContext,
        _dom: &Bool,
    ) -> bool
    {
        let u: f64 = rng.random();
        let threshold = 1.0 - scontext.bandwidth;
        if u < threshold {
            *x
        } else {
            !*x
        }
    }
}

pub enum MixedKernel {
    Gaussian(GaussianKernel),
    AitchisonAitken(AitchisonAitkenKernel),
}

#[derive(Serialize, Deserialize)]
pub enum MixedSContext {
    Gaussian(GaussianSContext),
    AitchisonAitken(AitchisonAitkenSContext),
}

#[derive(Serialize, Deserialize)]
pub enum MixedKContext {
    Gaussian(GaussianKContext),
    AitchisonAitken(()),
}

impl KernelFunc<Mixed> for MixedKernel 
{
    type SContext = MixedSContext;
    type KContext = MixedKContext;
    
    fn default_scontext(dom: &Mixed) -> Self::SContext {
        match dom {
            Mixed::Real(d) => MixedSContext::Gaussian(GaussianKernel::default_scontext(d)),
            Mixed::Nat(d) => MixedSContext::Gaussian(GaussianKernel::default_scontext(d)),
            Mixed::Int(d) => MixedSContext::Gaussian(GaussianKernel::default_scontext(d)),
            Mixed::Unit(d) => MixedSContext::Gaussian(GaussianKernel::default_scontext(d)),
            Mixed::Bool(d) => MixedSContext::AitchisonAitken(AitchisonAitkenKernel::default_scontext(d)),
            Mixed::Cat(d) => MixedSContext::AitchisonAitken(AitchisonAitkenKernel::default_scontext(d)),
            Mixed::GridReal(d) => MixedSContext::AitchisonAitken(AitchisonAitkenKernel::default_scontext(d)),
            Mixed::GridNat(d) => MixedSContext::AitchisonAitken(AitchisonAitkenKernel::default_scontext(d)),
            Mixed::GridInt(d) => MixedSContext::AitchisonAitken(AitchisonAitkenKernel::default_scontext(d)),
        }
    }
    
    fn update_scontext(bandwidth: f64, scontext: &mut Self::SContext, dom: &Mixed) {
        match (scontext, dom) {
            (MixedSContext::Gaussian(ctx), Mixed::Real(d)) => GaussianKernel::update_scontext(bandwidth, ctx, d),
            (MixedSContext::Gaussian(ctx), Mixed::Unit(d)) => GaussianKernel::update_scontext(bandwidth, ctx, d),
            (MixedSContext::Gaussian(ctx), Mixed::Int(d)) => GaussianKernel::update_scontext(bandwidth, ctx, d),
            (MixedSContext::Gaussian(ctx), Mixed::Nat(d)) => GaussianKernel::update_scontext(bandwidth, ctx, d),
            (MixedSContext::AitchisonAitken(ctx), Mixed::Bool(d)) => AitchisonAitkenKernel::update_scontext(bandwidth, ctx, d),
            (MixedSContext::AitchisonAitken(ctx), Mixed::Cat(d)) => AitchisonAitkenKernel::update_scontext(bandwidth, ctx, d),
            (MixedSContext::AitchisonAitken(ctx), Mixed::GridReal(d)) => AitchisonAitkenKernel::update_scontext(bandwidth, ctx, d),
            (MixedSContext::AitchisonAitken(ctx), Mixed::GridNat(d)) => AitchisonAitkenKernel::update_scontext(bandwidth, ctx, d),
            (MixedSContext::AitchisonAitken(ctx), Mixed::GridInt(d)) => AitchisonAitkenKernel::update_scontext(bandwidth, ctx, d),
            _ => panic!("Mismatched kernel context and input type"),
        }
    }
    
    fn default_kcontext(dom: &Mixed) -> Self::KContext {
        match dom {
            Mixed::Real(d) => MixedKContext::Gaussian(GaussianKernel::default_kcontext(d)),
            Mixed::Nat(d) => MixedKContext::Gaussian(GaussianKernel::default_kcontext(d)),
            Mixed::Int(d) => MixedKContext::Gaussian(GaussianKernel::default_kcontext(d)),
            Mixed::Unit(d) => MixedKContext::Gaussian(GaussianKernel::default_kcontext(d)),
            Mixed::Bool(_) => MixedKContext::AitchisonAitken(()),
            Mixed::Cat(_) => MixedKContext::AitchisonAitken(()),
            Mixed::GridReal(_) => MixedKContext::AitchisonAitken(()),
            Mixed::GridNat(_) => MixedKContext::AitchisonAitken(()),
            Mixed::GridInt(_) => MixedKContext::AitchisonAitken(()),
        }
    }
    
    fn update_kcontext(x: &MixedTypeDom, kcontext: &mut Self::KContext, scontext: &Self::SContext, dom: &Mixed) {
        match (x, kcontext, scontext, dom) {
            (MixedTypeDom::Real(x), MixedKContext::Gaussian(kctx), MixedSContext::Gaussian(sctx), Mixed::Real(d)) => GaussianKernel::update_kcontext(x, kctx, sctx, d),
            (MixedTypeDom::Unit(x), MixedKContext::Gaussian(kctx), MixedSContext::Gaussian(sctx), Mixed::Unit(d)) => GaussianKernel::update_kcontext(x, kctx, sctx, d),
            (MixedTypeDom::Int(x), MixedKContext::Gaussian(kctx), MixedSContext::Gaussian(sctx), Mixed::Int(d)) => GaussianKernel::update_kcontext(x, kctx, sctx, d),
            (MixedTypeDom::Nat(x), MixedKContext::Gaussian(kctx), MixedSContext::Gaussian(sctx), Mixed::Nat(d)) => GaussianKernel::update_kcontext(x, kctx, sctx, d),
            (MixedTypeDom::Bool(x), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::Bool(d)) => AitchisonAitkenKernel::update_kcontext(x, kctx, sctx, d),
            (MixedTypeDom::Cat(x), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::Cat(d)) => AitchisonAitkenKernel::update_kcontext(x, kctx, sctx, d),
            (MixedTypeDom::GridReal(x), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::GridReal(d)) => AitchisonAitkenKernel::update_kcontext(x, kctx, sctx, d),
            (MixedTypeDom::GridNat(x), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::GridNat(d)) => AitchisonAitkenKernel::update_kcontext(x, kctx, sctx, d),
            (MixedTypeDom::GridInt(x), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::GridInt(d)) => AitchisonAitkenKernel::update_kcontext(x, kctx, sctx, d),
            _ => panic!("Mismatched kernel context and input type"),
        }
    }
    
    fn compute(x1: &MixedTypeDom, x2: &MixedTypeDom, kcontext: &Self::KContext, scontext: &Self::SContext, dom: &Mixed) -> f64 {
        match (x1, x2, kcontext, scontext, dom) {
            (MixedTypeDom::Real(x), MixedTypeDom::Real(y), MixedKContext::Gaussian(kctx), MixedSContext::Gaussian(sctx), Mixed::Real(d)) => GaussianKernel::compute(x, y, kctx, sctx, d),
            (MixedTypeDom::Unit(x), MixedTypeDom::Unit(y), MixedKContext::Gaussian(kctx), MixedSContext::Gaussian(sctx), Mixed::Unit(d)) => GaussianKernel::compute(x, y, kctx, sctx, d),
            (MixedTypeDom::Int(x), MixedTypeDom::Int(y), MixedKContext::Gaussian(kctx), MixedSContext::Gaussian(sctx), Mixed::Int(d)) => GaussianKernel::compute(x, y, kctx, sctx, d),
            (MixedTypeDom::Nat(x), MixedTypeDom::Nat(y), MixedKContext::Gaussian(kctx), MixedSContext::Gaussian(sctx), Mixed::Nat(d)) => GaussianKernel::compute(x, y, kctx, sctx, d),
            (MixedTypeDom::Bool(x), MixedTypeDom::Bool(y), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::Bool(d)) => AitchisonAitkenKernel::compute(x, y, kctx, sctx, d),
            (MixedTypeDom::Cat(x), MixedTypeDom::Cat(y), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::Cat(d)) => AitchisonAitkenKernel::compute(x, y, kctx, sctx, d),
            (MixedTypeDom::GridReal(x), MixedTypeDom::GridReal(y), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::GridReal(d)) => AitchisonAitkenKernel::compute(x, y, kctx, sctx, d),
            (MixedTypeDom::GridNat(x), MixedTypeDom::GridNat(y), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::GridNat(d)) => AitchisonAitkenKernel::compute(x, y, kctx, sctx, d),
            (MixedTypeDom::GridInt(x), MixedTypeDom::GridInt(y), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::GridInt(d)) => AitchisonAitkenKernel::compute(x, y, kctx, sctx, d),
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
        context: &Self::KContext,
        scontext: &Self::SContext,
        dom: &Mixed,
    ) -> MixedTypeDom
    {
        match (x, context, scontext, dom) {
            (MixedTypeDom::Real(x), MixedKContext::Gaussian(kctx), MixedSContext::Gaussian(sctx), Mixed::Real(d)) => MixedTypeDom::Real(GaussianKernel::sample(rng, x, kctx, sctx, d)),
            (MixedTypeDom::Unit(x), MixedKContext::Gaussian(kctx), MixedSContext::Gaussian(sctx), Mixed::Unit(d)) => MixedTypeDom::Unit(GaussianKernel::sample(rng, x, kctx, sctx, d)),
            (MixedTypeDom::Int(x), MixedKContext::Gaussian(kctx), MixedSContext::Gaussian(sctx), Mixed::Int(d)) => MixedTypeDom::Int(GaussianKernel::sample(rng, x, kctx, sctx, d)),
            (MixedTypeDom::Nat(x), MixedKContext::Gaussian(kctx), MixedSContext::Gaussian(sctx), Mixed::Nat(d)) => MixedTypeDom::Nat(GaussianKernel::sample(rng, x, kctx, sctx, d)),
            (MixedTypeDom::Bool(x), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::Bool(d)) => MixedTypeDom::Bool(AitchisonAitkenKernel::sample(rng, x, kctx, sctx, d)),
            (MixedTypeDom::Cat(x), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::Cat(d)) => MixedTypeDom::Cat(AitchisonAitkenKernel::sample(rng, x, kctx, sctx, d)),
            (MixedTypeDom::GridReal(x), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::GridReal(d)) => MixedTypeDom::GridReal(AitchisonAitkenKernel::sample(rng, x, kctx, sctx, d)),
            (MixedTypeDom::GridNat(x), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::GridNat(d)) => MixedTypeDom::GridNat(AitchisonAitkenKernel::sample(rng, x, kctx, sctx, d)),
            (MixedTypeDom::GridInt(x), MixedKContext::AitchisonAitken(kctx), MixedSContext::AitchisonAitken(sctx), Mixed::GridInt(d)) => MixedTypeDom::GridInt(AitchisonAitkenKernel::sample(rng, x, kctx, sctx, d)),
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
    /// The shared context type for the kernel,
    /// which can hold precomputed values or parameters needed for efficient kernel computation across multiple calls. 
    /// This is useful for kernels that require expensive computations that can be reused, such as normalization constants or bandwidth parameters.
    type SContext: Serialize + for<'a> Deserialize<'a>;
    /// The context type for the kernel, 
    /// which can hold precomputed values or parameters needed 
    /// for efficient kernel computation.
    type KContext: Serialize + for<'a> Deserialize<'a>;


    fn default_scontext(scp: &Scp) -> Self::SContext;
    fn update_scontext(archive: &[Xy<S::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp);
    
    fn default_kcontext(scp: &Scp) -> Self::KContext;
    fn update_kcontext(archive: &[Xy<S::Raw, TypeCodom<Out>>], kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp);

    fn update_context(archive: &[Xy<S::Raw, TypeCodom<Out>>], kcontext: &mut[Self::KContext], scontext: &mut Self::SContext, scp: &Scp) {
        Self::update_scontext(archive, scontext, scp);
        Self::update_kcontext(archive, kcontext, scontext, scp);
    }

    fn compute(
        &self,
        s: &S::Raw,
        archive: &[Xy<S::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64;

    fn prior(&self, s: &S::Raw, scp: &Scp) -> f64;

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<S::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> S::Raw;
}

pub fn compute_bw(size: usize, dim: usize, dom: &Mixed) -> f64 {
    match dom {
        // Numerical
        Mixed::Real(d) => optuna_bw(size, dim, d),
        Mixed::Nat(d) => optuna_bw(size, dim, d),
        Mixed::Int(d) => optuna_bw(size, dim, d),
        Mixed::Unit(d) => optuna_bw(size, dim, d),
        // Categorical
        Mixed::Cat(d) => cat_bw(size, d),
        Mixed::GridReal(d) => cat_bw(size, d),
        Mixed::GridNat(d) => cat_bw(size, d),
        Mixed::GridInt(d) => cat_bw(size, d),
        Mixed::Bool(d) => cat_bw(size, d),
    }
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
    type KContext = Vec<MixedKContext>;
    type SContext = Vec<MixedSContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64
    {
        let product: f64 = s.iter().zip(scp.iter_opt()).enumerate()
        .map(
            |(d, (x1, dom))|
            {
                archive.iter()
                .zip(weights.weights.iter())
                .zip(kcontext.iter()).map(
                    |((comp, weight), kctx)|
                        MixedKernel::compute(x1, &comp.ref_x()[d], &kctx[d], &scontext[d], dom) * weight
                ).sum::<f64>()
            }
        ).product();
        let prior =
            <Univariate as Kernel<Mixed, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + product
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> <S>::Raw
    {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &kcontext[rng_idx];
                MixedKernel::sample(rng, &x[d], &ctx[d], &scontext[d], dom)
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

    fn default_scontext(scp: &Scp) -> Self::SContext {
        let dim = scp.size();
        (0..dim).map(|d| {
            let dom = scp.opt_at(d).unwrap();
            MixedKernel::default_scontext(dom)
        }).collect()
    }

    fn update_scontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp) {
        scp.iter_opt().zip(scontext.iter_mut()).for_each(|(dom, ctx)|
        {
            let bandwidth = compute_bw(archive.len(), scp.size(), dom);
            MixedKernel::update_scontext(bandwidth, ctx, dom);
        }
    )
    }

    fn default_kcontext(scp: &Scp) -> Self::KContext {
        scp.iter_opt().map(|dom| {
            MixedKernel::default_kcontext(dom)
        }).collect()
    }

    fn update_kcontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp) {
        archive.iter().zip(kcontext.iter_mut()).for_each(
            |(p, kctx)|
            p.ref_x().iter().zip(scp.iter_opt()).zip(kctx.iter_mut()).zip(scontext.iter()).for_each(
                |(((x, dom), ctx), sctx)|
                    MixedKernel::update_kcontext(x, ctx, sctx, dom)
            )
        );
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
    type KContext = Vec<GaussianKContext>;
    type SContext = Vec<GaussianSContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64
    {
        let product: f64 = s.iter().zip(scp.iter_opt()).enumerate()
        .map(
            |(d, (x1, dom))|
            {
                archive.iter()
                .zip(weights.weights.iter())
                .zip(kcontext.iter()).map(
                    |((comp, weight), kctx)|
                        GaussianKernel::compute(x1, &comp.ref_x()[d], &kctx[d], &scontext[d], dom) * weight
                ).sum::<f64>()
            }
        ).product();
        let prior =
            <Univariate as Kernel<Real, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + product
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> <S>::Raw
    {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &kcontext[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], &scontext[d], dom)
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

    fn default_scontext(scp: &Scp) -> Self::SContext {
        let dim = scp.size();
        (0..dim).map(|d| {
            let dom = scp.opt_at(d).unwrap();
            GaussianKernel::default_scontext(dom)
        }).collect()
    }

    fn update_scontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp) {
        scp.iter_opt().zip(scontext.iter_mut()).for_each(|(dom, ctx)|
        {
            let bandwidth = optuna_bw(archive.len(), scp.size(), dom);
            GaussianKernel::update_scontext(bandwidth, ctx, dom);
        }
    )
    }

    fn default_kcontext(scp: &Scp) -> Self::KContext {
        scp.iter_opt().map(|dom| {
            GaussianKernel::default_kcontext(dom)
        }).collect()
    }

    fn update_kcontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp) {
        archive.iter().zip(kcontext.iter_mut()).for_each(
            |(p, kctx)|
            p.ref_x().iter().zip(scp.iter_opt()).zip(kctx.iter_mut()).zip(scontext.iter()).for_each(
                |(((x, dom), ctx), sctx)|
                    GaussianKernel::update_kcontext(x, ctx, sctx, dom)
            )
        );
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
    type KContext = Vec<GaussianKContext>;
    type SContext = Vec<GaussianSContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64
    {
        let product: f64 = s.iter().zip(scp.iter_opt()).enumerate()
        .map(
            |(d, (x1, dom))|
            {
                archive.iter()
                .zip(weights.weights.iter())
                .zip(kcontext.iter()).map(
                    |((comp, weight), kctx)|
                        GaussianKernel::compute(x1, &comp.ref_x()[d], &kctx[d], &scontext[d], dom) * weight
                ).sum::<f64>()
            }
        ).product();
        let prior =
            <Univariate as Kernel<Int, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + product
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> <S>::Raw
    {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &kcontext[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], &scontext[d], dom)
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

    fn default_scontext(scp: &Scp) -> Self::SContext {
        let dim = scp.size();
        (0..dim).map(|d| {
            let dom = scp.opt_at(d).unwrap();
            GaussianKernel::default_scontext(dom)
        }).collect()
    }

    fn update_scontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp) {
        scp.iter_opt().zip(scontext.iter_mut()).for_each(|(dom, ctx)|
        {
            let bandwidth = optuna_bw(archive.len(), scp.size(), dom);
            GaussianKernel::update_scontext(bandwidth, ctx, dom);
        }
    )
    }

    fn default_kcontext(scp: &Scp) -> Self::KContext {
        scp.iter_opt().map(|dom| {
            GaussianKernel::default_kcontext(dom)
        }).collect()
    }

    fn update_kcontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp) {
        archive.iter().zip(kcontext.iter_mut()).for_each(
            |(p, kctx)|
            p.ref_x().iter().zip(scp.iter_opt()).zip(kctx.iter_mut()).zip(scontext.iter()).for_each(
                |(((x, dom), ctx), sctx)|
                    GaussianKernel::update_kcontext(x, ctx, sctx, dom)
            )
        );
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
    type KContext = Vec<GaussianKContext>;
    type SContext = Vec<GaussianSContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64
    {
        let product: f64 = s.iter().zip(scp.iter_opt()).enumerate()
        .map(
            |(d, (x1, dom))|
            {
                archive.iter()
                .zip(weights.weights.iter())
                .zip(kcontext.iter()).map(
                    |((comp, weight), kctx)|
                        GaussianKernel::compute(x1, &comp.ref_x()[d], &kctx[d], &scontext[d], dom) * weight
                ).sum::<f64>()
            }
        ).product();
        let prior =
            <Univariate as Kernel<Nat, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + product
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> <S>::Raw
    {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &kcontext[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], &scontext[d], dom)
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

    fn default_scontext(scp: &Scp) -> Self::SContext {
        let dim = scp.size();
        (0..dim).map(|d| {
            let dom = scp.opt_at(d).unwrap();
            GaussianKernel::default_scontext(dom)
        }).collect()
    }

    fn update_scontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp) {
        scp.iter_opt().zip(scontext.iter_mut()).for_each(|(dom, ctx)|
        {
            let bandwidth = optuna_bw(archive.len(), scp.size(), dom);
            GaussianKernel::update_scontext(bandwidth, ctx, dom);
        }
    )
    }

    fn default_kcontext(scp: &Scp) -> Self::KContext {
        scp.iter_opt().map(|dom| {
            GaussianKernel::default_kcontext(dom)
        }).collect()
    }

    fn update_kcontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp) {
        archive.iter().zip(kcontext.iter_mut()).for_each(
            |(p, kctx)|
            p.ref_x().iter().zip(scp.iter_opt()).zip(kctx.iter_mut()).zip(scontext.iter()).for_each(
                |(((x, dom), ctx), sctx)|
                    GaussianKernel::update_kcontext(x, ctx, sctx, dom)
            )
        );
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
        type KContext = Vec<GaussianKContext>;
    type SContext = Vec<GaussianSContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64
    {
        let product: f64 = s.iter().zip(scp.iter_opt()).enumerate()
        .map(
            |(d, (x1, dom))|
            {
                archive.iter()
                .zip(weights.weights.iter())
                .zip(kcontext.iter()).map(
                    |((comp, weight), kctx)|
                        GaussianKernel::compute(x1, &comp.ref_x()[d], &kctx[d], &scontext[d], dom) * weight
                ).sum::<f64>()
            }
        ).product();
        let prior =
            <Univariate as Kernel<Unit, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + product
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> <S>::Raw
    {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &kcontext[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], &scontext[d], dom)
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

    fn default_scontext(scp: &Scp) -> Self::SContext {
        let dim = scp.size();
        (0..dim).map(|d| {
            let dom = scp.opt_at(d).unwrap();
            GaussianKernel::default_scontext(dom)
        }).collect()
    }

    fn update_scontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp) {
        scp.iter_opt().zip(scontext.iter_mut()).for_each(|(dom, ctx)|
        {
            let bandwidth = optuna_bw(archive.len(), scp.size(), dom);
            GaussianKernel::update_scontext(bandwidth, ctx, dom);
        }
    )
    }

    fn default_kcontext(scp: &Scp) -> Self::KContext {
        scp.iter_opt().map(|dom| {
            GaussianKernel::default_kcontext(dom)
        }).collect()
    }

    fn update_kcontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp) {
        archive.iter().zip(kcontext.iter_mut()).for_each(
            |(p, kctx)|
            p.ref_x().iter().zip(scp.iter_opt()).zip(kctx.iter_mut()).zip(scontext.iter()).for_each(
                |(((x, dom), ctx), sctx)|
                    GaussianKernel::update_kcontext(x, ctx, sctx, dom)
            )
        );
    }
}

impl<T, Scp, S, SolId, SInfo, Out> Kernel<GridDom<T>, Scp, S, SolId, SInfo, Out> for Univariate
where
    T: GridBounds,
    Scp: Searchspace<S, SolId, SInfo, Opt = GridDom<T>> + HasVariables,
    S: Uncomputed<SolId, GridDom<T>, SInfo, Raw = Arc<[TypeDom<GridDom<T>>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type KContext = ();
    type SContext = Vec<AitchisonAitkenSContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        _kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64
    {
        let product: f64 = s.iter().zip(scp.iter_opt()).enumerate()
        .map(
            |(d, (x1, dom))|
            {
                archive.iter()
                .zip(weights.weights.iter())
                .map(
                    |(comp, weight)|
                        AitchisonAitkenKernel::compute(x1, &comp.ref_x()[d], &(), &scontext[d], dom) * weight
                ).sum::<f64>()
            }
        ).product();
        let prior =
            <Univariate as Kernel<GridDom<T>, Scp, S, SolId, SInfo, Out>>::prior(self, s, scp);
        weights.prior_weight * prior + product
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        _kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> <S>::Raw
    {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                AitchisonAitkenKernel::sample(rng, &x[d], &(), &scontext[d], dom)
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

    fn default_scontext(scp: &Scp) -> Self::SContext {
        let dim = scp.size();
        (0..dim).map(|d| {
            let dom = scp.opt_at(d).unwrap();
            AitchisonAitkenKernel::default_scontext(dom)
        }).collect()
    }

    fn update_scontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp) {
        scp.iter_opt().zip(scontext.iter_mut()).for_each(|(dom, ctx)|
        {
            let bandwidth = cat_bw(archive.len(), dom);
            AitchisonAitkenKernel::update_scontext(bandwidth, ctx, dom);
        }
    )
    }

    fn default_kcontext(scp: &Scp) -> Self::KContext {
        scp.iter_opt().map(|dom| {
            AitchisonAitkenKernel::default_kcontext(dom)
        }).collect()
    }

    fn update_kcontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], _kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp) {
        archive.iter().for_each(
            |p|
            p.ref_x().iter().zip(scp.iter_opt()).zip(scontext.iter()).for_each(
                |((x, dom), sctx)|
                    AitchisonAitkenKernel::update_kcontext(x, &mut (), sctx, dom)
            )
        );
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
    type KContext = Vec<MixedKContext>;
    type SContext = Vec<MixedSContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64
    {
        let sum: f64 = archive.iter()
        .zip(weights.weights.iter())
        .zip(kcontext.iter()).map(
            |((comp, weight), ctx)|
            {
                s.iter().zip(comp.ref_x().iter()).zip(scp.iter_opt()).zip(ctx.iter()).zip(scontext.iter()).map(
                    |((((x1, x2),dom), kctx), sctx)|
                    {
                        MixedKernel::compute(x1, x2, kctx, sctx, dom)
                    }
                ).product::<f64>() * weight
            }
        ).sum();
        let prior = <Multivariate as Kernel<Mixed, Scp, S, SolId, SInfo, Out>>::prior(self,s,scp);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> <S>::Raw
    {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &kcontext[rng_idx];
                MixedKernel::sample(rng, &x[d], &ctx[d], &scontext[d], dom)
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

    fn default_scontext(scp: &Scp) -> Self::SContext {
        let dim = scp.size();
        (0..dim).map(|d| {
            let dom = scp.opt_at(d).unwrap();
            MixedKernel::default_scontext(dom)
        }).collect()
    }

    fn update_scontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp) {
        scp.iter_opt().zip(scontext.iter_mut()).for_each(|(dom, ctx)|
        {
            let bandwidth = compute_bw(archive.len(), scp.size(), dom);
            MixedKernel::update_scontext(bandwidth, ctx, dom);
        }
    )
    }

    fn default_kcontext(scp: &Scp) -> Self::KContext {
        scp.iter_opt().map(|dom| {
            MixedKernel::default_kcontext(dom)
        }).collect()
    }

    fn update_kcontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp) {
        archive.iter().zip(kcontext.iter_mut()).for_each(
            |(p, kctx)|
            p.ref_x().iter().zip(scp.iter_opt()).zip(kctx.iter_mut()).zip(scontext.iter()).for_each(
                |(((x, dom), ctx), sctx)|
                    MixedKernel::update_kcontext(x, ctx, sctx, dom)
            )
        );
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
    type KContext = Vec<GaussianKContext>;
    type SContext = Vec<GaussianSContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64
    {
        let sum: f64 = archive.iter()
        .zip(weights.weights.iter())
        .zip(kcontext.iter()).map(
            |((comp, weight), ctx)|
            {
                s.iter().zip(comp.ref_x().iter()).zip(scp.iter_opt()).zip(ctx.iter()).zip(scontext.iter()).map(
                    |((((x1, x2),dom), kctx), sctx)|
                    {
                        GaussianKernel::compute(x1, x2, kctx, sctx, dom)
                    }
                ).product::<f64>() * weight
            }
        ).sum();
        let prior = <Multivariate as Kernel<Real, Scp, S, SolId, SInfo, Out>>::prior(self,s,scp);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> <S>::Raw
    {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &kcontext[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], &scontext[d], dom)
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

    fn default_scontext(scp: &Scp) -> Self::SContext {
        let dim = scp.size();
        (0..dim).map(|d| {
            let dom = scp.opt_at(d).unwrap();
            GaussianKernel::default_scontext(dom)
        }).collect()
    }

    fn update_scontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp) {
        scp.iter_opt().zip(scontext.iter_mut()).for_each(|(dom, ctx)|
        {
            let bandwidth = optuna_bw(archive.len(), scp.size(), dom);
            GaussianKernel::update_scontext(bandwidth, ctx, dom);
        }
    )
    }

    fn default_kcontext(scp: &Scp) -> Self::KContext {
        scp.iter_opt().map(|dom| {
            GaussianKernel::default_kcontext(dom)
        }).collect()
    }

    fn update_kcontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp) {
        archive.iter().zip(kcontext.iter_mut()).for_each(
            |(p, kctx)|
            p.ref_x().iter().zip(scp.iter_opt()).zip(kctx.iter_mut()).zip(scontext.iter()).for_each(
                |(((x, dom), ctx), sctx)|
                    GaussianKernel::update_kcontext(x, ctx, sctx, dom)
            )
        );
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
    type KContext = Vec<GaussianKContext>;
    type SContext = Vec<GaussianSContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64
    {
        let sum: f64 = archive.iter()
        .zip(weights.weights.iter())
        .zip(kcontext.iter()).map(
            |((comp, weight), ctx)|
            {
                s.iter().zip(comp.ref_x().iter()).zip(scp.iter_opt()).zip(ctx.iter()).zip(scontext.iter()).map(
                    |((((x1, x2),dom), kctx), sctx)|
                    {
                        GaussianKernel::compute(x1, x2, kctx, sctx, dom)
                    }
                ).product::<f64>() * weight
            }
        ).sum();
        let prior = <Multivariate as Kernel<Int, Scp, S, SolId, SInfo, Out>>::prior(self,s,scp);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> <S>::Raw
    {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &kcontext[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], &scontext[d], dom)
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

    fn default_scontext(scp: &Scp) -> Self::SContext {
        let dim = scp.size();
        (0..dim).map(|d| {
            let dom = scp.opt_at(d).unwrap();
            GaussianKernel::default_scontext(dom)
        }).collect()
    }

    fn update_scontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp) {
        scp.iter_opt().zip(scontext.iter_mut()).for_each(|(dom, ctx)|
        {
            let bandwidth = optuna_bw(archive.len(), scp.size(), dom);
            GaussianKernel::update_scontext(bandwidth, ctx, dom);
        }
    )
    }

    fn default_kcontext(scp: &Scp) -> Self::KContext {
        scp.iter_opt().map(|dom| {
            GaussianKernel::default_kcontext(dom)
        }).collect()
    }

    fn update_kcontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp) {
        archive.iter().zip(kcontext.iter_mut()).for_each(
            |(p, kctx)|
            p.ref_x().iter().zip(scp.iter_opt()).zip(kctx.iter_mut()).zip(scontext.iter()).for_each(
                |(((x, dom), ctx), sctx)|
                    GaussianKernel::update_kcontext(x, ctx, sctx, dom)
            )
        );
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
    type KContext = Vec<GaussianKContext>;
    type SContext = Vec<GaussianSContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64
    {
        let sum: f64 = archive.iter()
        .zip(weights.weights.iter())
        .zip(kcontext.iter()).map(
            |((comp, weight), ctx)|
            {
                s.iter().zip(comp.ref_x().iter()).zip(scp.iter_opt()).zip(ctx.iter()).zip(scontext.iter()).map(
                    |((((x1, x2),dom), kctx), sctx)|
                    {
                        GaussianKernel::compute(x1, x2, kctx, sctx, dom)
                    }
                ).product::<f64>() * weight
            }
        ).sum();
        let prior = <Multivariate as Kernel<Nat, Scp, S, SolId, SInfo, Out>>::prior(self,s,scp);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> <S>::Raw
    {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &kcontext[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], &scontext[d], dom)
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

    fn default_scontext(scp: &Scp) -> Self::SContext {
        let dim = scp.size();
        (0..dim).map(|d| {
            let dom = scp.opt_at(d).unwrap();
            GaussianKernel::default_scontext(dom)
        }).collect()
    }

    fn update_scontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp) {
        scp.iter_opt().zip(scontext.iter_mut()).for_each(|(dom, ctx)|
        {
            let bandwidth = optuna_bw(archive.len(), scp.size(), dom);
            GaussianKernel::update_scontext(bandwidth, ctx, dom);
        }
    )
    }

    fn default_kcontext(scp: &Scp) -> Self::KContext {
        scp.iter_opt().map(|dom| {
            GaussianKernel::default_kcontext(dom)
        }).collect()
    }

    fn update_kcontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp) {
        archive.iter().zip(kcontext.iter_mut()).for_each(
            |(p, kctx)|
            p.ref_x().iter().zip(scp.iter_opt()).zip(kctx.iter_mut()).zip(scontext.iter()).for_each(
                |(((x, dom), ctx), sctx)|
                    GaussianKernel::update_kcontext(x, ctx, sctx, dom)
            )
        );
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
    type KContext = Vec<GaussianKContext>;
    type SContext = Vec<GaussianSContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64
    {
        let sum: f64 = archive.iter()
        .zip(weights.weights.iter())
        .zip(kcontext.iter()).map(
            |((comp, weight), ctx)|
            {
                s.iter().zip(comp.ref_x().iter()).zip(scp.iter_opt()).zip(ctx.iter()).zip(scontext.iter()).map(
                    |((((x1, x2),dom), kctx), sctx)|
                    {
                        GaussianKernel::compute(x1, x2, kctx, sctx, dom)
                    }
                ).product::<f64>() * weight
            }
        ).sum();
        let prior = <Multivariate as Kernel<Unit, Scp, S, SolId, SInfo, Out>>::prior(self,s,scp);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> <S>::Raw
    {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                let ctx = &kcontext[rng_idx];
                GaussianKernel::sample(rng, &x[d], &ctx[d], &scontext[d], dom)
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

    fn default_scontext(scp: &Scp) -> Self::SContext {
        let dim = scp.size();
        (0..dim).map(|d| {
            let dom = scp.opt_at(d).unwrap();
            GaussianKernel::default_scontext(dom)
        }).collect()
    }

    fn update_scontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp) {
        scp.iter_opt().zip(scontext.iter_mut()).for_each(|(dom, ctx)|
        {
            let bandwidth = optuna_bw(archive.len(), scp.size(), dom);
            GaussianKernel::update_scontext(bandwidth, ctx, dom);
        }
    )
    }

    fn default_kcontext(scp: &Scp) -> Self::KContext {
        scp.iter_opt().map(|dom| {
            GaussianKernel::default_kcontext(dom)
        }).collect()
    }

    fn update_kcontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp) {
        archive.iter().zip(kcontext.iter_mut()).for_each(
            |(p, kctx)|
            p.ref_x().iter().zip(scp.iter_opt()).zip(kctx.iter_mut()).zip(scontext.iter()).for_each(
                |(((x, dom), ctx), sctx)|
                    GaussianKernel::update_kcontext(x, ctx, sctx, dom)
            )
        );
    }
}

impl<T, Scp, S, SolId, SInfo, Out> Kernel<GridDom<T>, Scp, S, SolId, SInfo, Out> for Multivariate
where
    T: GridBounds,
    Scp: Searchspace<S, SolId, SInfo, Opt = GridDom<T>> + HasVariables,
    S: Uncomputed<SolId, GridDom<T>, SInfo, Raw = Arc<[TypeDom<GridDom<T>>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type KContext = ();
    type SContext = Vec<AitchisonAitkenSContext>;

    fn compute(
        &self,
        s: &<S>::Raw,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        _kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        weights: &PointWeights,
        scp: &Scp,
    ) -> f64
    {
        let sum: f64 = archive.iter()
        .zip(weights.weights.iter())
        .map(
            |(comp, weight)|
            {
                s.iter().zip(comp.ref_x().iter()).zip(scp.iter_opt()).zip(scontext.iter()).map(
                    |(((x1, x2),dom), sctx)|
                    {
                        AitchisonAitkenKernel::compute(x1, x2, &(), sctx, dom)
                    }
                ).product::<f64>() * weight
            }
        ).sum();
        let prior = <Multivariate as Kernel<GridDom<T>, Scp, S, SolId, SInfo, Out>>::prior(self,s,scp);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<<S>::Raw, TypeCodom<Out>>],
        _kcontext: &[Self::KContext],
        scontext: &Self::SContext,
        scp: &Scp,
    ) -> <S>::Raw
    {
        let dim = scp.size();
        (0..dim)
            .map(|d| {
                let dom = scp.opt_at(d).unwrap();
                let rng_idx = rng.random_range(0..archive.len());
                let x = archive[rng_idx].ref_x();
                AitchisonAitkenKernel::sample(rng, &x[d], &(), &scontext[d], dom)
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

    fn default_scontext(scp: &Scp) -> Self::SContext {
        let dim = scp.size();
        (0..dim).map(|d| {
            let dom = scp.opt_at(d).unwrap();
            AitchisonAitkenKernel::default_scontext(dom)
        }).collect()
    }

    fn update_scontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], scontext: &mut Self::SContext, scp: &Scp) {
        scp.iter_opt().zip(scontext.iter_mut()).for_each(|(dom, ctx)|
        {
            let bandwidth = cat_bw(archive.len(), dom);
            AitchisonAitkenKernel::update_scontext(bandwidth, ctx, dom);
        }
    )
    }

    fn default_kcontext(scp: &Scp) -> Self::KContext {
        scp.iter_opt().map(|dom| {
            AitchisonAitkenKernel::default_kcontext(dom)
        }).collect()
    }

    fn update_kcontext(archive: &[Xy<<S>::Raw, TypeCodom<Out>>], _kcontext: &mut[Self::KContext], scontext: &Self::SContext, scp: &Scp) {
        archive.iter().for_each(
            |p|
            p.ref_x().iter().zip(scp.iter_opt()).zip(scontext.iter()).for_each(
                |((x, dom), sctx)|
                    AitchisonAitkenKernel::update_kcontext(x, &mut (), sctx, dom)
            )
        );
    }
}
