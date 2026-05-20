
use tantale_core::{Codomain, Computed, Domain, GridDom, HasX, Id, Int, Mixed, MixedTypeDom, Nat, Outcome, Real, Searchspace, SolInfo, SolutionShape, Uncomputed, Unit, Xy, domain::{CategoricalDomain, NumericalDomain, TypeDom, grid::GridBounds}, has_trait::HasVariables};

use serde::{Deserialize, Serialize};
use statrs::{distribution::{Continuous, ContinuousCDF, Normal as StatrsNormal }, function::erf::erf};
use rand_distr::{Distribution, Normal};
use rand::{Rng, RngExt, seq::IndexedRandom};
use std::{f64::consts::PI, sync::Arc};
use num::{Num, cast::AsPrimitive};

use crate::bayesian::{bandwidth::{cat_bw, optuna_bw}, weighter::PointWeights};

pub trait KernelFunc<Dom: Domain> {
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
    fn compute(&self, x1: &Dom::TypeDom, x2: &Dom::TypeDom, bandwidth: f64, dom: &Dom) -> f64;

    /// Computes the prior probability for a given point `x` in the domain `dom`. 
    fn prior(&self, x: &Dom::TypeDom, dom: &Dom) -> f64;
    
    /// Samples a value from the kernel distribution at `x` with the given `bandwidth` and domain `dom`.
    fn sample<R: Rng>(&self, rng: &mut R, x: &Dom::TypeDom, bandwidth: f64, dom: &Dom) -> Dom::TypeDom;
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

impl GaussianKernel {
    /// Computes :
    /// $$
    /// \\begin{split}
    ///     \\mathbb{P}\\left( L < X = x_2 < U \\right) &= \\int_{L}^{U} K(x,x_2\\,|\\,b)dx \\\\
    ///                 &= \\frac{1}{2}\\left(\\text{erf}(\\frac{(R - x_2)}{(\\sqrt{2} b)}) -\\text{erf}(\\frac{(L - x_2)}{(\\sqrt{2} b)}) \\right)
    /// \\end{split}
    /// $$ 
    pub fn gaussian_interval<T>(x:&T, bandwidth: f64, low: f64, up: f64) -> f64
    where
        T: Num + AsPrimitive<f64>,
    {
        let sqrt_2 = 2.0_f64.sqrt();
        let x_f = x.as_();
        
        let up_erf  = erf((up-x_f)/(sqrt_2*bandwidth));
        let low_erf  = erf((low-x_f)/(sqrt_2*bandwidth));
        (up_erf - low_erf) / 2.0
    }
}

impl KernelFunc<Real> for GaussianKernel
{
    fn compute(&self, x1: &f64, x2: &f64, bandwidth: f64, dom: &Real) -> f64
    {
        let lhs = 1. / (bandwidth * (2.0 * PI).sqrt());
        let (low, up) = dom.get_ref_bounds();
        let cst = Self::gaussian_interval(x2, bandwidth, low.as_(), up.as_());
        lhs * (-0.5 * ((x1 - x2)/bandwidth).powi(2)).exp() / cst
    }
    
    fn prior(&self, x: &f64, dom: &Real) -> f64 {
        let (low, up) = dom.get_bounds();
        let mean = (low + up) / 2.0;
        let std = up - low;
        StatrsNormal::new(mean, std).unwrap().pdf(*x)
    }
    
    /// Samples a value from the truncated Gaussian distribution at `x` with the given 
    /// `bandwidth` and domain `dom`.
    /// 
    /// # Notes
    /// 
    /// The sampling is performed using the inverse transform sampling method, 
    /// which involves sampling a uniform random variable and applying the inverse CDF 
    /// of the truncated Gaussian distribution.
    fn sample<R: Rng>(&self, rng: &mut R, x: &f64, bandwidth: f64, dom: &Real) -> f64 {
        let (low, up) = dom.get_ref_bounds();
        let normal = StatrsNormal::new(*x, bandwidth).unwrap();
        let p_low = normal.cdf(*low);
        let p_up  = normal.cdf(*up);
        if (p_up - p_low).abs() < f64::EPSILON {
            return *x;
        }
        let u: f64 = rng.random_range(p_low..p_up);
        normal.inverse_cdf(u)
    }
}

impl KernelFunc<Unit> for GaussianKernel
{
    fn compute(&self, x1: &f64, x2: &f64, bandwidth: f64, _dom: &Unit) -> f64
    {
        let lhs = 1. / (bandwidth * (2.0 * PI).sqrt());
        let cst = Self::gaussian_interval(x2, bandwidth, 0.0, 1.0);
        lhs * (-0.5 * ((x1 - x2)/bandwidth).powi(2)).exp() / cst
    }
    
    fn prior(&self, x: &f64, _dom: &Unit) -> f64 {
        StatrsNormal::new(0.5, 1.0).unwrap().pdf(*x)
    }

    /// Samples a value from the truncated Gaussian distribution at `x` with the given 
    /// `bandwidth` and domain `dom`.
    /// 
    /// # Notes
    /// 
    /// The sampling is performed using the inverse transform sampling method, 
    /// which involves sampling a uniform random variable and applying the inverse CDF 
    /// of the truncated Gaussian distribution.
    fn sample<R: Rng>(&self, rng: &mut R, x: &f64, bandwidth: f64, _dom: &Unit) -> f64 {
        let normal = StatrsNormal::new(*x, bandwidth).unwrap();
        let p_low = normal.cdf(0.0);
        let p_up  = normal.cdf(1.0);
        if (p_up - p_low).abs() < f64::EPSILON {
            return *x;
        }
        let u: f64 = rng.random_range(p_low..p_up);
        normal.inverse_cdf(u)
    }
}

impl KernelFunc<Int> for GaussianKernel
{
    fn compute(&self, x1: &i64, x2: &i64, bandwidth: f64, dom: &Int) -> f64
    {
        let (low, up) = dom.get_ref_bounds();
        let low =  *low as f64 - 0.5;
        let up =  *up  as f64 + 0.5;
        let cst = Self::gaussian_interval(x2, bandwidth, low, up);
        let low = *x1 as f64 - 0.5;
        let up = *x1 as f64 + 0.5;
        let cdf = Self::gaussian_interval(x2, bandwidth, low, up);
        cdf / cst
    }
    
    fn prior(&self, x: &i64, dom: &Int) -> f64 {
        let (low, up) = dom.get_bounds();
        let low: f64 = low.as_();
        let up: f64 = up.as_();
        let mean = (low + up) / 2.0;
        let std = up - low;
        StatrsNormal::new(mean, std).unwrap().pdf(x.as_())
    }

    fn sample<R: Rng>(&self, rng: &mut R, x: &i64, bandwidth: f64, dom: &Int) -> i64 {
        let (low, up) = dom.get_ref_bounds();
        let distr = Normal::new(x.as_(), bandwidth).unwrap();
        (distr.sample(rng).round() as i64).clamp(*low, *up)
    }
}

impl KernelFunc<Nat> for GaussianKernel
{
    fn compute(&self, x1: &u64, x2: &u64, bandwidth: f64, dom: &Nat) -> f64
    {
        let (low, up) = dom.get_ref_bounds();
        let low =  *low as f64 - 0.5;
        let up =  *up  as f64 + 0.5;
        let cst = Self::gaussian_interval(x2, bandwidth, low, up);
        let low = *x1 as f64 - 0.5;
        let up = *x1 as f64 + 0.5;
        let cdf = Self::gaussian_interval(x2, bandwidth, low, up);
        cdf / cst
    }
    
        fn prior(&self, x: &u64, dom: &Nat) -> f64 {
        let (low, up) = dom.get_bounds();
        let low: f64 = low.as_();
        let up: f64 = up.as_();
        let mean = (low + up) / 2.0;
        let std = up - low;
        StatrsNormal::new(mean, std).unwrap().pdf(x.as_())
    }

    fn sample<R: Rng>(&self, rng: &mut R, x: &u64, bandwidth: f64, dom: &Nat) -> u64 {
        let (low, up) = dom.get_ref_bounds();
        let distr = Normal::new(x.as_(), bandwidth).unwrap();
        (distr.sample(rng).round() as u64).clamp(*low, *up)
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

impl<C:CategoricalDomain> KernelFunc<C> for AitchisonAitkenKernel
{
    fn compute(&self, x1: &C::TypeDom, x2: &C::TypeDom,  bandwidth: f64, dom: &C) -> f64
    {
        if x1 == x2 {
            1.0 - bandwidth
        } else {
            bandwidth / (dom.size() as f64 - 1.0)
        }
    }

    fn prior(&self, _x: &<C as Domain>::TypeDom, dom: &C) -> f64 {
        1.0 / (dom.size() as f64)
    }
    
    fn sample<R: Rng>(&self, rng: &mut R, x: &TypeDom<C>, bandwidth: f64, dom: &C) -> TypeDom<C> {
        let u: f64 = rng.random();
        let threshold = 1.0 - bandwidth;
        if u < threshold {
            x.clone()
        } else {
            // Sample a different category than x with equal probability
            let other_categories: Vec<&C::TypeDom> = dom.get_features().iter().filter(|&c| c != x).collect();
            let idx = rng.random_range(0..other_categories.len());
            other_categories[idx].clone()
        }
    }
}

pub enum MixedKernel {
    Gaussian(GaussianKernel),
    AitchisonAitken(AitchisonAitkenKernel),   
}

impl KernelFunc<Mixed> for MixedKernel
{
    fn compute(&self, x1: &TypeDom<Mixed>, x2: &TypeDom<Mixed>, bandwidth: f64, dom: &Mixed) -> f64 {
        match (x1, x2, dom) {
            // Numerical
            (MixedTypeDom::Real(r1), MixedTypeDom::Real(r2), Mixed::Real(d)) => {
                GaussianKernel.compute(r1, r2, bandwidth, d)
            },
            (MixedTypeDom::Unit(u1), MixedTypeDom::Unit(u2), Mixed::Unit(d)) => {
                GaussianKernel.compute(u1, u2, bandwidth, d)
            },
            (MixedTypeDom::Int(i1), MixedTypeDom::Int(i2), Mixed::Int(d)) => {
                GaussianKernel.compute(i1, i2, bandwidth, d)
            },
            (MixedTypeDom::Nat(n1), MixedTypeDom::Nat(n2), Mixed::Nat(d)) => {
                GaussianKernel.compute(n1, n2, bandwidth, d)
            },
            // Categorical
            (MixedTypeDom::Cat(c1), MixedTypeDom::Cat(c2), Mixed::Cat(d)) => {
                AitchisonAitkenKernel.compute(c1, c2, bandwidth, d)
            },
            (MixedTypeDom::GridInt(i1), MixedTypeDom::GridInt(i2), Mixed::GridInt(d)) => {
                AitchisonAitkenKernel.compute(i1, i2, bandwidth, d)
            },
            (MixedTypeDom::GridNat(n1), MixedTypeDom::GridNat(n2), Mixed::GridNat(d)) => {
                AitchisonAitkenKernel.compute(n1, n2, bandwidth, d)
            },
            (MixedTypeDom::GridReal(r1), MixedTypeDom::GridReal(r2), Mixed::GridReal(d)) => {
                AitchisonAitkenKernel.compute(r1, r2, bandwidth, d)
            },
            _ => panic!("Mismatched types in Mixed kernel computation"),
        }
    }
    
    fn sample<R: Rng>(&self, rng: &mut R, x: &TypeDom<Mixed>, bandwidth: f64, dom: &Mixed) -> TypeDom<Mixed> {
        match (x, dom) {
            // Numerical
            (MixedTypeDom::Real(x), Mixed::Real(d)) => {
                MixedTypeDom::Real(GaussianKernel.sample(rng, x, bandwidth, d))
            },
            (MixedTypeDom::Unit(x), Mixed::Unit(d)) => {
                MixedTypeDom::Unit(GaussianKernel.sample(rng, x, bandwidth, d))
            },
            (MixedTypeDom::Int(x), Mixed::Int(d)) => {
                MixedTypeDom::Int(GaussianKernel.sample(rng, x, bandwidth, d))
            },
            (MixedTypeDom::Nat(x), Mixed::Nat(d)) => {
                MixedTypeDom::Nat(GaussianKernel.sample(rng, x, bandwidth, d))
            },
            // Categorical
            (MixedTypeDom::Cat(x), Mixed::Cat(d)) => {
                MixedTypeDom::Cat(AitchisonAitkenKernel.sample(rng, x, bandwidth, d))
            },
            (MixedTypeDom::GridInt(x), Mixed::GridInt(d)) => {
                MixedTypeDom::GridInt(AitchisonAitkenKernel.sample(rng, x, bandwidth, d))
            },
            (MixedTypeDom::GridNat(x), Mixed::GridNat(d)) => {
                MixedTypeDom::GridNat(AitchisonAitkenKernel.sample(rng, x, bandwidth, d))
            },
            (MixedTypeDom::GridReal(x), Mixed::GridReal(d)) => {
                MixedTypeDom::GridReal(AitchisonAitkenKernel.sample(rng, x, bandwidth, d))
            },
            _ => panic!("Mismatched types in Mixed kernel computation"),
        }
    }
    
    fn prior(&self, x: &TypeDom<Mixed>, dom: &Mixed) -> f64 {
        match (x, dom) {
            // Numerical
            (MixedTypeDom::Real(x), Mixed::Real(d)) => {
                GaussianKernel.prior(x, d)
            },
            (MixedTypeDom::Unit(x), Mixed::Unit(d)) => {
                GaussianKernel.prior(x, d)
            },
            (MixedTypeDom::Int(x), Mixed::Int(d)) => {
                GaussianKernel.prior(x, d)
            },
            (MixedTypeDom::Nat(x), Mixed::Nat(d)) => {
                GaussianKernel.prior(x, d)
            },
            // Categorical
            (MixedTypeDom::Cat(x), Mixed::Cat(d)) => {
                AitchisonAitkenKernel.prior(x, d)
            },
            (MixedTypeDom::GridInt(x), Mixed::GridInt(d)) => {
                AitchisonAitkenKernel.prior(x, d)
            },
            (MixedTypeDom::GridNat(x), Mixed::GridNat(d)) => {
                AitchisonAitkenKernel.prior(x, d)
            },
            (MixedTypeDom::GridReal(x), Mixed::GridReal(d)) => {
                AitchisonAitkenKernel.prior(x, d)
            },
            _ => panic!("Mismatched types in Mixed kernel computation"),
        }
    }
}

/// Trait for kernel functions used in the TPE algorithm. 
/// The kernel function computes the similarity between a given solution and the points in the archive, which is used to estimate the density of good and bad solutions.
/// 
/// # See also
/// - [`Univariate`] for the univariate kernel, which assumes independence between dimensions and computes the product of the kernel values for each dimension.
/// - [`Multivariate`] for the multivariate kernel, which models the joint distribution of all dimensions.
pub trait Kernel<D, Sp, S, SolId, SInfo, Cod, Out>
where
    Self: Sized + Serialize + for<'a> Deserialize<'a>,
    D: Domain,
    Sp: Searchspace<S, SolId, SInfo, Opt=D>,
    S: Uncomputed<SolId, D, SInfo>,
    S::Twin<Sp::Obj>: Uncomputed<SolId, Sp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{

    type Context;

    fn prepare(
        &self,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        searchspace: &Sp
    ) -> Self::Context;

    fn compute(
        &self,
        s: &S::Raw, 
        archive: &[Xy<S::Raw, Cod::TypeCodom>], 
        weights: &PointWeights,
        searchspace: &Sp, 
        context: &Self::Context
    ) -> f64;

    fn prior(&self, s: &S::Raw, searchspace: &Sp) -> f64;

    fn sample<R: Rng>(
        &self, 
        rng: &mut R, 
        archive: &[Xy<S::Raw, Cod::TypeCodom>], 
        searchspace: &Sp, 
        context: &Self::Context
    ) -> S::Raw;
}

pub fn compute_bw(size: usize, dim: usize, dom: &Mixed) -> f64
{
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

pub fn compute_kernel(x1: &TypeDom<Mixed>, x2: &TypeDom<Mixed>, bandwidth: f64, dom: &Mixed) -> f64
{
    match (x1, x2, dom) {
        // Numerical
        (MixedTypeDom::Real(r1), MixedTypeDom::Real(r2), Mixed::Real(dom)) => {
            GaussianKernel.compute(r1, r2, bandwidth, dom)
        },
        (MixedTypeDom::Unit(u1), MixedTypeDom::Unit(u2), Mixed::Unit(dom)) => {
            GaussianKernel.compute(u1, u2, bandwidth, dom)
        },
        (MixedTypeDom::Int(i1), MixedTypeDom::Int(i2), Mixed::Int(dom)) => {
            GaussianKernel.compute(i1, i2, bandwidth, dom)
        },
        (MixedTypeDom::Nat(n1), MixedTypeDom::Nat(n2), Mixed::Nat(dom)) => {
            GaussianKernel.compute(n1, n2, bandwidth, dom)
        },
        // Categorical
        (MixedTypeDom::Cat(c1), MixedTypeDom::Cat(c2), Mixed::Cat(dom)) => {
            AitchisonAitkenKernel.compute(c1, c2, bandwidth, dom)
        },
        (MixedTypeDom::Bool(b1), MixedTypeDom::Bool(b2), Mixed::Bool(dom)) => {
            AitchisonAitkenKernel.compute(b1, b2, bandwidth, dom)
        },
        (MixedTypeDom::GridInt(i1), MixedTypeDom::GridInt(i2), Mixed::GridInt(dom)) => {
            AitchisonAitkenKernel.compute(i1, i2, bandwidth, dom)
        },
        (MixedTypeDom::GridNat(n1), MixedTypeDom::GridNat(n2), Mixed::GridNat(dom)) => {
            AitchisonAitkenKernel.compute(n1, n2, bandwidth, dom)
        },
        (MixedTypeDom::GridReal(r1), MixedTypeDom::GridReal(r2), Mixed::GridReal(dom)) => {
            AitchisonAitkenKernel.compute(r1, r2, bandwidth, dom)
        },
        _ => panic!("Mismatched types in Univariate kernel computation"),
    }
}

pub fn sample_kernel<R: Rng>(rng: &mut R, x: &TypeDom<Mixed>, bandwidth: f64, dom: &Mixed) -> TypeDom<Mixed>
{
    match (x, dom) {
        // Numerical
        (MixedTypeDom::Real(r1), Mixed::Real(dom)) => {
            MixedTypeDom::Real(GaussianKernel.sample(rng, r1, bandwidth, dom))
        },
        (MixedTypeDom::Unit(u1), Mixed::Unit(dom)) => {
            MixedTypeDom::Unit(GaussianKernel.sample(rng, u1, bandwidth, dom))
        },
        (MixedTypeDom::Int(i1), Mixed::Int(dom)) => {
            MixedTypeDom::Int(GaussianKernel.sample(rng, i1, bandwidth, dom))
        },
        (MixedTypeDom::Nat(n1), Mixed::Nat(dom)) => {
            MixedTypeDom::Nat(GaussianKernel.sample(rng, n1, bandwidth, dom))
        },
        // Categorical
        (MixedTypeDom::Cat(c1), Mixed::Cat(dom)) => {
            MixedTypeDom::Cat(AitchisonAitkenKernel.sample(rng, c1, bandwidth, dom))
        },
        (MixedTypeDom::Bool(b1), Mixed::Bool(dom)) => {
            MixedTypeDom::Bool(AitchisonAitkenKernel.sample(rng, b1, bandwidth, dom))
        },
        (MixedTypeDom::GridInt(i1), Mixed::GridInt(dom)) => {
            MixedTypeDom::GridInt(AitchisonAitkenKernel.sample(rng, i1, bandwidth, dom))
        },
        (MixedTypeDom::GridNat(n1), Mixed::GridNat(dom)) => {
            MixedTypeDom::GridNat(AitchisonAitkenKernel.sample(rng, n1, bandwidth, dom))
        },
        (MixedTypeDom::GridReal(r1), Mixed::GridReal(dom)) => {
            MixedTypeDom::GridReal(AitchisonAitkenKernel.sample(rng, r1, bandwidth, dom))
        },
        _ => panic!("Mismatched types in Univariate kernel computation"),
    }
}

pub fn compute_prior(x: &TypeDom<Mixed>, dom: &Mixed) -> f64
{
    match (x, dom) {
        (MixedTypeDom::Real(r1), Mixed::Real(dom)) => {
            GaussianKernel.prior(r1, dom)
        },
        (MixedTypeDom::Unit(u1), Mixed::Unit(dom)) => {
            GaussianKernel.prior(u1, dom)
        },
        (MixedTypeDom::Int(i1), Mixed::Int(dom)) => {
            GaussianKernel.prior(i1, dom)
        },
        (MixedTypeDom::Nat(n1), Mixed::Nat(dom)) => {
            GaussianKernel.prior(n1, dom)
        },
        (MixedTypeDom::Cat(c1), Mixed::Cat(dom)) => {
            AitchisonAitkenKernel.prior(c1, dom)
        },
        (MixedTypeDom::Bool(b1), Mixed::Bool(dom)) => {
            AitchisonAitkenKernel.prior(b1, dom)
        },
        (MixedTypeDom::GridInt(i1), Mixed::GridInt(dom)) => {
            AitchisonAitkenKernel.prior(i1, dom)
        },
        (MixedTypeDom::GridNat(n1), Mixed::GridNat(dom)) => {
            AitchisonAitkenKernel.prior(n1, dom)
        },
        (MixedTypeDom::GridReal(r1), Mixed::GridReal(dom)) => {
            AitchisonAitkenKernel.prior(r1, dom)
        },
        _ => panic!("Mismatched types in Univariate kernel computation"),
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

impl<Scp, S, SolId, SInfo, Cod, Out> Kernel<Mixed, Scp, S, SolId, SInfo, Cod, Out> for Univariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Mixed> + HasVariables,
    S: Uncomputed<SolId, Mixed, SInfo, Raw = Arc<[TypeDom<Mixed>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    type Context = Vec<f64>;

    fn compute(
        &self,
        s1: &S::Raw,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        weights: &PointWeights,
        searchspace: &Scp,
        context: &Self::Context
    ) -> f64
    {
        let mut product = 1.0;
        for (d, x1) in s1.iter().enumerate() {
            let mut sum = 0.0;
            let dom = searchspace.opt_at(d).unwrap();
            let bandwidth = context[d];
            for (comp, weight) in archive.iter().zip(weights.weights.iter()) {
                let s2 = comp.ref_x();
                let x2 = &s2[d];
                sum += compute_kernel(x1, x2, bandwidth, dom) * weight;
            }
            product *= sum;
        }
        let prior = <Univariate as Kernel<Mixed, Scp, S, SolId, SInfo, Cod, Out>>::prior(self, s1, searchspace);
        weights.prior_weight * prior + product
    }
    
    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        searchspace: &Scp,
        context: &Self::Context
    ) -> S::Raw {
        let dim = searchspace.size();
        (0..dim).map(
            |d|
            {
                let dom = searchspace.opt_at(d).unwrap();
                let bandwidth = context[d];
                let x = archive.choose(rng).unwrap().ref_x();
                sample_kernel(rng, &x[d], bandwidth, dom)
            }
        ).collect()
    }
    
    fn prior(&self, s: &S::Raw, searchspace: &Scp) -> f64 {
        s.iter().enumerate().map(|(d, x)| {
            let dom = searchspace.opt_at(d).unwrap();
            compute_prior(x, dom)
        }).product()
    }
    
    fn prepare(&self, archive: &[Xy<S::Raw, Cod::TypeCodom>], searchspace: &Scp) -> Self::Context {
        let size = archive.len();
        let dim = searchspace.size();
        (0..searchspace.size()).map(|d| {
            let dom = searchspace.opt_at(d).unwrap();
            compute_bw(size, dim, dom)
        }).collect()
    }
}

impl<Scp, S, SolId, SInfo, Cod, Out> Kernel<Real, Scp, S, SolId, SInfo, Cod, Out> for Univariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Real> + HasVariables,
    S: Uncomputed<SolId, Real, SInfo, Raw = Arc<[TypeDom<Real>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Xy<S::Raw, Cod::TypeCodom>: SolutionShape<SolId,SInfo, SolOpt = Computed<S, SolId, Real, Cod, Out, SInfo>>
{
    type Context = Vec<f64>;
    fn compute(
        &self,
        s1: &S::Raw,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        weights: &PointWeights,
        searchspace: &Scp,
        context: &Self::Context
    ) -> f64
    {
        let mut product = 1.0;
        for (d, x1) in s1.iter().enumerate() {
            let mut sum = 0.0;
            let dom = searchspace.opt_at(d).unwrap();
            let bandwidth = context[d];
            for (comp, weight) in archive.iter().zip(weights.weights.iter()) {
                let s2 = comp.ref_x();
                let x2 = &s2[d];
                sum += GaussianKernel.compute(x1, x2, bandwidth, dom) * weight;
            }
            product *= sum;
        }
        let prior = self.prior(s1, searchspace);
        weights.prior_weight * prior + product
    }
    
    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        searchspace: &Scp,
        context: &Self::Context
    ) -> S::Raw {
        let dim = searchspace.size();
        (0..dim).map(
            |d|
            {
                let dom = searchspace.opt_at(d).unwrap();
                let bandwidth = context[d];
                let x = archive.choose(rng).unwrap().ref_x();
                GaussianKernel.sample(rng, &x[d], bandwidth, dom)
            }
        ).collect()
    }
    
    fn prior(&self, s: &S::Raw, searchspace: &Scp) -> f64 {
        s.iter().enumerate().map(|(d, x)| {
            let dom = searchspace.opt_at(d).unwrap();
            GaussianKernel.prior(x, dom)
        }).product()
    }
    
    fn prepare(&self, archive: &[Xy<S::Raw, Cod::TypeCodom>], searchspace: &Scp) -> Self::Context {
        let size = archive.len();
        let dim = searchspace.size();
        (0..searchspace.size()).map(|d| {
            let dom = searchspace.opt_at(d).unwrap();
            optuna_bw(size, dim, dom)
        }).collect()
    }
}

impl<Scp, S, SolId, SInfo, Cod, Out> Kernel<Int, Scp, S, SolId, SInfo, Cod, Out> for Univariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Int> + HasVariables,
    S: Uncomputed<SolId, Int, SInfo, Raw = Arc<[TypeDom<Int>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Xy<S::Raw, Cod::TypeCodom>: SolutionShape<SolId,SInfo, SolOpt = Computed<S, SolId, Int, Cod, Out, SInfo>>
{
    type Context = Vec<f64>;
    fn compute(
        &self,
        s1: &S::Raw,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        weights: &PointWeights,
        searchspace: &Scp,
        context: &Self::Context
    ) -> f64
    {
        let mut product = 1.0;
        for (d, x1) in s1.iter().enumerate() {
            let mut sum = 0.0;
            let dom = searchspace.opt_at(d).unwrap();
            let bandwidth = context[d];
            for (comp, weight) in archive.iter().zip(weights.weights.iter()) {
                let s2 = comp.ref_x();
                let x2 = &s2[d];
                sum += GaussianKernel.compute(x1, x2, bandwidth, dom) * weight;
            }
            product *= sum;
        }
        let prior = self.prior(s1, searchspace);
        weights.prior_weight * prior + product
    }
    
    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        searchspace: &Scp,
        context: &Self::Context
    ) -> S::Raw {
        let dim = searchspace.size();
        (0..dim).map(
            |d|
            {
                let dom = searchspace.opt_at(d).unwrap();
                let bandwidth = context[d];
                let x = archive.choose(rng).unwrap().get_sopt().ref_x();
                GaussianKernel.sample(rng, &x[d], bandwidth, dom)
            }
        ).collect()
    }

    fn prior(&self, s: &S::Raw, searchspace: &Scp) -> f64 {
        s.iter().enumerate().map(|(d, x)| {
            let dom = searchspace.opt_at(d).unwrap();
            GaussianKernel.prior(x, dom)
        }).product()
    }

    fn prepare(&self, archive: &[Xy<S::Raw, Cod::TypeCodom>], searchspace: &Scp) -> Self::Context {
        let size = archive.len();
        let dim = searchspace.size();
        (0..searchspace.size()).map(|d| {
            let dom = searchspace.opt_at(d).unwrap();
            optuna_bw(size, dim, dom)
        }).collect()
    }
}

impl<Scp, S, SolId, SInfo, Cod, Out> Kernel<Nat, Scp, S, SolId, SInfo, Cod, Out> for Univariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Nat> + HasVariables,
    S: Uncomputed<SolId, Nat, SInfo, Raw = Arc<[TypeDom<Nat>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Xy<S::Raw, Cod::TypeCodom>: SolutionShape<SolId,SInfo, SolOpt = Computed<S, SolId, Nat, Cod, Out, SInfo>>
{
    type Context = Vec<f64>;

    fn compute(
        &self,
        s1: &S::Raw,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        weights: &PointWeights,
        searchspace: &Scp,
        context: &Self::Context
    ) -> f64
    {
        let mut product = 1.0;
        for (d, x1) in s1.iter().enumerate() {
            let mut sum = 0.0;
            let dom = searchspace.opt_at(d).unwrap();
            let bandwidth = context[d];
            for (comp, weight) in archive.iter().zip(weights.weights.iter()) {
                let s2 = comp.ref_x();
                let x2 = &s2[d];
                sum += GaussianKernel.compute(x1, x2, bandwidth, dom) * weight;
            }
            product *= sum;
        }
        let prior = self.prior(s1, searchspace);
        weights.prior_weight * prior + product
    }
    
    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        searchspace: &Scp,
        context: &Self::Context
    ) -> S::Raw {
        let dim = searchspace.size();
        (0..dim).map(
            |d|
            {
                let dom = searchspace.opt_at(d).unwrap();
                let bandwidth = context[d];
                let x = archive.choose(rng).unwrap().get_sopt().ref_x();
                GaussianKernel.sample(rng, &x[d], bandwidth, dom)
            }
        ).collect()
    }

    fn prior(&self, s: &S::Raw, searchspace: &Scp) -> f64 {
        s.iter().enumerate().map(|(d, x)| {
            let dom = searchspace.opt_at(d).unwrap();
            GaussianKernel.prior(x, dom)
        }).product()
    }

    fn prepare(&self, archive: &[Xy<S::Raw, Cod::TypeCodom>], searchspace: &Scp) -> Self::Context {
        let size = archive.len();
        let dim = searchspace.size();
        (0..searchspace.size()).map(|d| {
            let dom = searchspace.opt_at(d).unwrap();
            optuna_bw(size, dim, dom)
        }).collect()
    }
}

impl<Scp, S, SolId, SInfo, Cod, Out> Kernel<Unit, Scp, S, SolId, SInfo, Cod, Out> for Univariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Unit> + HasVariables,
    S: Uncomputed<SolId, Unit, SInfo, Raw = Arc<[TypeDom<Unit>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Xy<S::Raw, Cod::TypeCodom>: SolutionShape<SolId,SInfo, SolOpt = Computed<S, SolId, Unit, Cod, Out, SInfo>>
{
    type Context = Vec<f64>;
    fn compute(
        &self,
        s1: &S::Raw,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        weights: &PointWeights,
        searchspace: &Scp,
        context: &Self::Context
    ) -> f64
    {
        let mut product = 1.0;
        for (d, x1) in s1.iter().enumerate() {
            let mut sum = 0.0;
            let dom = searchspace.opt_at(d).unwrap();
            let bandwidth = context[d];
            for (comp, weight) in archive.iter().zip(weights.weights.iter()) {
                let s2 = comp.ref_x();
                let x2 = &s2[d];
                sum += GaussianKernel.compute(x1, x2, bandwidth, dom) * weight;
            }
            product *= sum;
        }
        let prior = self.prior(s1, searchspace);
        weights.prior_weight * prior + product
    }
    
    fn sample<R: Rng>(
        &self, rng: &mut R,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        searchspace: &Scp,
        context: &Self::Context
    ) -> S::Raw {
        let dim = searchspace.size();
        (0..dim).map(
            |d|
            {
                let dom = searchspace.opt_at(d).unwrap();
                let bandwidth = context[d];
                let x = archive.choose(rng).unwrap().get_sopt().ref_x();
                GaussianKernel.sample(rng, &x[d], bandwidth, dom)
            }
        ).collect()
    }

    fn prior(&self, s: &S::Raw, searchspace: &Scp) -> f64 {
        s.iter().enumerate().map(|(d, x)| {
            let dom = searchspace.opt_at(d).unwrap();
            GaussianKernel.prior(x, dom)
        }).product()
    }

    fn prepare(&self, archive: &[Xy<S::Raw, Cod::TypeCodom>], searchspace: &Scp) -> Self::Context {
        let size = archive.len();
        let dim = searchspace.size();
        (0..searchspace.size()).map(|d| {
            let dom = searchspace.opt_at(d).unwrap();
            optuna_bw(size, dim, dom)
        }).collect()
    }
}

impl<T, Scp, S, SolId, SInfo, Cod, Out> Kernel<GridDom<T>, Scp, S, SolId, SInfo, Cod, Out> for Univariate 
where
    T: GridBounds,
    Scp: Searchspace<S, SolId, SInfo, Opt=GridDom<T>> + HasVariables,
    S: Uncomputed<SolId, GridDom<T>, SInfo, Raw = Arc<[TypeDom<GridDom<T>>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Xy<S::Raw, Cod::TypeCodom>: SolutionShape<SolId,SInfo, SolOpt = Computed<S, SolId, GridDom<T>, Cod, Out, SInfo>>
{
    type Context = Vec<f64>;
    fn compute(
        &self,
        s1: &S::Raw,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        weights: &PointWeights,
        searchspace: &Scp,
        context: &Self::Context
    ) -> f64
    {
        let mut product = 1.0;
        for (d, x1) in s1.iter().enumerate() {
            let mut sum = 0.0;
            let dom = searchspace.opt_at(d).unwrap();
            let bandwidth = context[d];
            for (comp, weight) in archive.iter().zip(weights.weights.iter()) {
                let s2 = comp.ref_x();
                let x2 = &s2[d];
                sum += AitchisonAitkenKernel.compute(x1, x2, bandwidth, dom) * weight;
            }
            product *= sum;
        }
        let prior = self.prior(s1, searchspace);
        weights.prior_weight * prior + product
    }
    
    fn sample<R: Rng>(
        &self,
        rng: &mut R,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        searchspace: &Scp,
        context: &Self::Context
    ) -> S::Raw {
        let dim = searchspace.size();
        (0..dim).map(
            |d|
            {
                let dom = searchspace.opt_at(d).unwrap();
                let bandwidth = context[d];
                let x = archive.choose(rng).unwrap().get_sopt().ref_x();
                AitchisonAitkenKernel.sample(rng, &x[d], bandwidth, dom)
            }
        ).collect()
    }

    fn prior(&self, s: &S::Raw, searchspace: &Scp) -> f64 {
        s.iter().enumerate().map(|(d, x)| {
            let dom = searchspace.opt_at(d).unwrap();
            AitchisonAitkenKernel.prior(x, dom)
        }).product()
    }

    fn prepare(&self, archive: &[Xy<S::Raw, Cod::TypeCodom>], searchspace: &Scp) -> Self::Context {
        let size = archive.len();
        (0..searchspace.size()).map(|d| {
            let dom = searchspace.opt_at(d).unwrap();
            cat_bw(size, dom)
        }).collect()
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

impl<Scp, S, SolId, SInfo, Cod, Out> Kernel<Mixed, Scp, S, SolId, SInfo, Cod, Out> for Multivariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Mixed> + HasVariables,
    S: Uncomputed<SolId, Mixed, SInfo, Raw = Arc<[TypeDom<Mixed>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    type Context = Vec<f64>;
    fn compute(
        &self,
        s1: &S::Raw,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        weights: &PointWeights,
        searchspace: &Scp,
        context: &Self::Context
    ) -> f64
    {
        let mut sum = 0.0;
        for (comp, weight) in archive.iter().zip(weights.weights.iter()) {
            let s2 = comp.ref_x();
            let mut product = 1.0;
            for (d, x1) in s1.iter().enumerate() {
                let dom = searchspace.opt_at(d).unwrap();
                let b = context[d];
                let x2 = &s2[d];
                product *= compute_kernel(x1, x2, b, dom);
            }
            sum += product * weight;
        }
        let prior = <Multivariate as Kernel<Mixed, Scp, S, SolId, SInfo, Cod, Out>>::prior(self, s1, searchspace);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self, 
        rng: &mut R, 
        archive: &[Xy<S::Raw, Cod::TypeCodom>], 
        searchspace: &Scp, 
        context: &Self::Context
    ) -> S::Raw {
        let dim = searchspace.size();
        let x  = archive.choose(rng).unwrap().ref_x();
        (0..dim).map(
            |d|
            {
                let dom = searchspace.opt_at(d).unwrap();
                let bandwidth = context[d];
                sample_kernel(rng, &x[d], bandwidth, dom)
            }
        ).collect()
    }

    fn prior(&self, s: &S::Raw, searchspace: &Scp) -> f64 {
        s.iter().enumerate().map(|(d, x)| {
            let dom = searchspace.opt_at(d).unwrap();
            compute_prior(x, dom)
        }).product()
    }

    fn prepare(&self, archive: &[Xy<S::Raw, Cod::TypeCodom>], searchspace: &Scp) -> Self::Context {
        let size = archive.len();
        let dim = searchspace.size();
        (0..searchspace.size()).map(|d| {
            let dom = searchspace.opt_at(d).unwrap();
            compute_bw(size, dim, dom)
        }).collect()
    }
}

impl<Scp, S, SolId, SInfo, Cod, Out> Kernel<Real, Scp, S, SolId, SInfo, Cod, Out> for Multivariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Real> + HasVariables,
    S: Uncomputed<SolId, Real, SInfo, Raw = Arc<[TypeDom<Real>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Xy<S::Raw, Cod::TypeCodom>: SolutionShape<SolId,SInfo, SolOpt = Computed<S, SolId, Real, Cod, Out, SInfo>>
{

    type Context = Vec<f64>;

    fn compute(
        &self,
        s1: &S::Raw,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        weights: &PointWeights,
        searchspace: &Scp,
        context: &Self::Context
    ) -> f64
    {
        let mut sum = 0.0;
        for (comp, weight) in archive.iter().zip(weights.weights.iter()) {
            let s2 = comp.ref_x();
            let mut product = 1.0;
            for (d, x1) in s1.iter().enumerate() {
                let dom = searchspace.opt_at(d).unwrap();
                let b = context[d];
                let x2 = &s2[d];
                product *= GaussianKernel.compute(x1, x2, b, dom);
            }
            sum += product * weight;
        }
        let prior = self.prior(s1, searchspace);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self, 
        rng: &mut R, 
        archive: &[Xy<S::Raw, Cod::TypeCodom>], 
        searchspace: &Scp,
        context: &Self::Context
    ) -> S::Raw {
        let dim = searchspace.size();
        let x  = archive.choose(rng).unwrap().get_sopt().ref_x();
        (0..dim).map(
            |d|
            {
                let dom = searchspace.opt_at(d).unwrap();
                let bandwidth = context[d];
                GaussianKernel.sample(rng, &x[d], bandwidth, dom)
            }
        ).collect()
    }

    fn prior(&self, s: &S::Raw, searchspace: &Scp) -> f64 {
        s.iter().enumerate().map(|(d, x)| {
            let dom = searchspace.opt_at(d).unwrap();
            GaussianKernel.prior(x, dom)
        }).product()
    }

    fn prepare(&self, archive: &[Xy<S::Raw, Cod::TypeCodom>], searchspace: &Scp) -> Self::Context {
        let size = archive.len();
        let dim = searchspace.size();
        (0..searchspace.size()).map(|d| {
            let dom = searchspace.opt_at(d).unwrap();
            optuna_bw(size, dim, dom)
        }).collect()
    }
}

impl<Scp, S, SolId, SInfo, Cod, Out> Kernel<Int, Scp, S, SolId, SInfo, Cod, Out> for Multivariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Int> + HasVariables,
    S: Uncomputed<SolId, Int, SInfo, Raw = Arc<[TypeDom<Int>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Xy<S::Raw, Cod::TypeCodom>: SolutionShape<SolId,SInfo, SolOpt = Computed<S, SolId, Int, Cod, Out, SInfo>>
{
    type Context = Vec<f64>;
    fn compute(
        &self,
        s1: &S::Raw,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        weights: &PointWeights,
        searchspace: &Scp,
        context: &Self::Context
    ) -> f64
    {
        let mut sum = 0.0;
        for (comp, weight) in archive.iter().zip(weights.weights.iter()) {
            let s2 = comp.ref_x();
            let mut product = 1.0;
            for (d, x1) in s1.iter().enumerate() {
                let dom = searchspace.opt_at(d).unwrap();
                let b = context[d];
                let x2 = &s2[d];
                product *= GaussianKernel.compute(x1, x2, b, dom);
            }
            sum += product * weight;
        }
        let prior = self.prior(s1, searchspace);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self, 
        rng: &mut R, 
        archive: &[Xy<S::Raw, Cod::TypeCodom>], 
        searchspace: &Scp,
        context: &Self::Context
    ) -> S::Raw {
        let dim = searchspace.size();
        let x  = archive.choose(rng).unwrap().get_sopt().ref_x();
        (0..dim).map(
            |d|
            {
                let dom = searchspace.opt_at(d).unwrap();
                let bandwidth = context[d];
                GaussianKernel.sample(rng, &x[d], bandwidth, dom)
            }
        ).collect()
    }

    fn prior(&self, s: &S::Raw, searchspace: &Scp) -> f64 {
        s.iter().enumerate().map(|(d, x)| {
            let dom = searchspace.opt_at(d).unwrap();
            GaussianKernel.prior(x, dom)
        }).product()
    }

    fn prepare(&self, archive: &[Xy<S::Raw, Cod::TypeCodom>], searchspace: &Scp) -> Self::Context {
        let size = archive.len();
        let dim = searchspace.size();
        (0..searchspace.size()).map(|d| {
            let dom = searchspace.opt_at(d).unwrap();
            optuna_bw(size, dim, dom)
        }).collect()
    }
}

impl<Scp, S, SolId, SInfo, Cod, Out> Kernel<Nat, Scp, S, SolId, SInfo, Cod, Out> for Multivariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Nat> + HasVariables,
    S: Uncomputed<SolId, Nat, SInfo, Raw = Arc<[TypeDom<Nat>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Xy<S::Raw, Cod::TypeCodom>: SolutionShape<SolId,SInfo, SolOpt = Computed<S, SolId, Nat, Cod, Out, SInfo>>
{
    type Context = Vec<f64>;
    fn compute(
        &self,
        s1: &S::Raw,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        weights: &PointWeights,
        searchspace: &Scp,
        context: &Self::Context
    ) -> f64
    {
        let mut sum = 0.0;
        for (comp, weight) in archive.iter().zip(weights.weights.iter()) {
            let s2 = comp.ref_x();
            let mut product = 1.0;
            for (d, x1) in s1.iter().enumerate() {
                let dom = searchspace.opt_at(d).unwrap();
                let b = context[d];
                let x2 = &s2[d];
                product *= GaussianKernel.compute(x1, x2, b, dom);
            }
            sum += product * weight;
        }
        let prior = self.prior(s1, searchspace);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self, 
        rng: &mut R, 
        archive: &[Xy<S::Raw, Cod::TypeCodom>], 
        searchspace: &Scp,
        context: &Self::Context
    ) -> S::Raw {
        let dim = searchspace.size();
        let x  = archive.choose(rng).unwrap().get_sopt().ref_x();
        (0..dim).map(
            |d|
            {
                let dom = searchspace.opt_at(d).unwrap();
                let bandwidth = context[d];
                GaussianKernel.sample(rng, &x[d], bandwidth, dom)
            }
        ).collect()
    }

    fn prior(&self, s: &S::Raw, searchspace: &Scp) -> f64 {
        s.iter().enumerate().map(|(d, x)| {
            let dom = searchspace.opt_at(d).unwrap();
            GaussianKernel.prior(x, dom)
        }).product()
    }

    fn prepare(&self, archive: &[Xy<S::Raw, Cod::TypeCodom>], searchspace: &Scp) -> Self::Context {
        let size = archive.len();
        let dim = searchspace.size();
        (0..searchspace.size()).map(|d| {
            let dom = searchspace.opt_at(d).unwrap();
            optuna_bw(size, dim, dom)
        }).collect()
    }
}

impl<Scp, S, SolId, SInfo, Cod, Out> Kernel<Unit, Scp, S, SolId, SInfo, Cod, Out> for Multivariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Unit> + HasVariables,
    S: Uncomputed<SolId, Unit, SInfo, Raw = Arc<[TypeDom<Unit>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Xy<S::Raw, Cod::TypeCodom>: SolutionShape<SolId,SInfo, SolOpt = Computed<S, SolId, Unit, Cod, Out, SInfo>>
{
    type Context = Vec<f64>;

    fn compute(
        &self,
        s1: &S::Raw,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        weights: &PointWeights,
        searchspace: &Scp,
        context: &Self::Context
    ) -> f64
    {
        let mut sum = 0.0;
        for (comp, weight) in archive.iter().zip(weights.weights.iter()) {
            let s2 = comp.ref_x();
            let mut product = 1.0;
            for (d, x1) in s1.iter().enumerate() {
                let dom = searchspace.opt_at(d).unwrap();
                let b = context[d];
                let x2 = &s2[d];
                product *= GaussianKernel.compute(x1, x2, b, dom);
            }
            sum += product * weight;
        }
        let prior = self.prior(s1, searchspace);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self, 
        rng: &mut R, 
        archive: &[Xy<S::Raw, Cod::TypeCodom>], 
        searchspace: &Scp,
        context: &Self::Context
    ) -> S::Raw {
        let dim = searchspace.size();
        let x  = archive.choose(rng).unwrap().get_sopt().ref_x();
        (0..dim).map(
            |d|
            {
                let dom = searchspace.opt_at(d).unwrap();
                let bandwidth = context[d];
                GaussianKernel.sample(rng, &x[d], bandwidth, dom)
            }
        ).collect()
    }

    fn prior(&self, s: &S::Raw, searchspace: &Scp) -> f64 {
        s.iter().enumerate().map(|(d, x)| {
            let dom = searchspace.opt_at(d).unwrap();
            GaussianKernel.prior(x, dom)
        }).product()
    }

    fn prepare(&self, archive: &[Xy<S::Raw, Cod::TypeCodom>], searchspace: &Scp) -> Self::Context {
        let size = archive.len();
        let dim = searchspace.size();
        (0..searchspace.size()).map(|d| {
            let dom = searchspace.opt_at(d).unwrap();
            optuna_bw(size, dim, dom)
        }).collect()
    }
}

impl<T, Scp, S, SolId, SInfo, Cod, Out> Kernel<GridDom<T>, Scp, S, SolId, SInfo, Cod, Out> for Multivariate 
where
    T: GridBounds,
    Scp: Searchspace<S, SolId, SInfo, Opt=GridDom<T>> + HasVariables,
    S: Uncomputed<SolId, GridDom<T>, SInfo, Raw = Arc<[TypeDom<GridDom<T>>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Xy<S::Raw, Cod::TypeCodom>: SolutionShape<SolId,SInfo, SolOpt = Computed<S, SolId, GridDom<T>, Cod, Out, SInfo>>
{
    type Context = Vec<f64>;

    fn compute(
        &self,
        s1: &S::Raw,
        archive: &[Xy<S::Raw, Cod::TypeCodom>],
        weights: &PointWeights,
        searchspace: &Scp,
        context: &Self::Context
    ) -> f64
    {
        let mut sum = 0.0;
        for (comp, weight) in archive.iter().zip(weights.weights.iter()) {
            let s2 = comp.ref_x();
            let mut product = 1.0;
            for (d, x1) in s1.iter().enumerate() {
                let dom = searchspace.opt_at(d).unwrap();
                let b = context[d];
                let x2 = &s2[d];
                product *= AitchisonAitkenKernel.compute(x1, x2, b, dom);
            }
            sum += product * weight;
        }
        let prior = self.prior(s1, searchspace);
        weights.prior_weight * prior + sum
    }

    fn sample<R: Rng>(
        &self, 
        rng: &mut R, 
        archive: &[Xy<S::Raw, Cod::TypeCodom>], 
        searchspace: &Scp,
        context: &Self::Context
    ) -> S::Raw {
        let dim = searchspace.size();
        let x = archive.choose(rng).unwrap().get_sopt().ref_x();
        (0..dim).map(
            |d|
            {
                let dom = searchspace.opt_at(d).unwrap();
                let bandwidth = context[d];
                AitchisonAitkenKernel.sample(rng, &x[d], bandwidth, dom)
            }
        ).collect()
    }

    fn prior(&self, s: &S::Raw, searchspace: &Scp) -> f64 {
        s.iter().enumerate().map(|(d, x)| {
            let dom = searchspace.opt_at(d).unwrap();
            AitchisonAitkenKernel.prior(x, dom)
        }).product()
    }

    fn prepare(&self, archive: &[Xy<S::Raw, Cod::TypeCodom>], searchspace: &Scp) -> Self::Context {
        let size = archive.len();
        (0..searchspace.size()).map(|d| {
            let dom = searchspace.opt_at(d).unwrap();
            cat_bw(size, dom)
        }).collect()
    }
}