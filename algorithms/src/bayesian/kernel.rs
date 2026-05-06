
use tantale_core::{Domain, GridDom, Id, Int, LinkOpt, Mixed, MixedTypeDom, Nat, Real, Searchspace, SolInfo, Solution, Unit, domain::{CategoricalDomain, NumericalDomain, TypeDom, grid::GridBounds}, has_trait::HasVariables};

use serde::{Deserialize, Serialize};
use statrs::function::erf::erf;
use rand_distr::{Distribution, Normal};
use rand::Rng;
use std::{f64::consts::PI, sync::Arc};
use num::{Num, cast::AsPrimitive};

use crate::bayesian::bandwidth::{cat_bw, optuna_bw};

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

    /// Samples a value from the kernel distribution at `x` with the given `bandwidth` and domain `dom`.
    fn sample<R: Rng>(&self, rng: &mut R, x: &Dom::TypeDom, bandwidth: f64, dom: &Dom) -> f64;
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
        
        let up_erf  =erf((up-x_f)/(sqrt_2*bandwidth));
        let low_erf  =erf((low-x_f)/(sqrt_2*bandwidth));
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
        lhs * (-0.5 * ((x1 - x2)/bandwidth).powi(2)) / cst
    }
    
    fn sample<R: Rng>(&self, rng: &mut R, x: &TypeDom<Real>, bandwidth: f64, dom: &Real) -> f64 {
        let (low, up) = dom.get_ref_bounds();
        let distr = Normal::new(*x, bandwidth).unwrap();
        distr.sample(rng).clamp(*low, *up)
    }
}

impl KernelFunc<Unit> for GaussianKernel
{
    fn compute(&self, x1: &f64, x2: &f64, bandwidth: f64, _dom: &Unit) -> f64
    {
        let lhs = 1. / (bandwidth * (2.0 * PI).sqrt());
        let cst = Self::gaussian_interval(x2, bandwidth, 0.0, 1.0);
        lhs * (-0.5 * ((x1 - x2)/bandwidth).powi(2)) / cst
    }
    
    fn sample<R: Rng>(&self, rng: &mut R, x: &<Unit as Domain>::TypeDom, bandwidth: f64, dom: &Unit) -> f64 {
        todo!()
    }
}

impl KernelFunc<Int> for GaussianKernel
{
    /// Computes the Gaussian kernel value between two integer points, taking into account the bounds of the domain.
    /// The kernel value is computed as
    /// $$
    /// \\begin{split}
    ///     K(x_1, x_2) &= \\int_{x_1 - \frac{1}{2}}^{x_1 + \frac{1}{2}} K(x,x_2\\,|\\,b)dx \\\\
    ///                 &= \\frac{1}{2}\\left(\\text{erf}(\\frac{(\\frac{1}{2} - x_2 + x_1)}{(\\sqrt{2} b)}) -\\text{erf}(\\frac{(x_1 - \\frac{1}{2} - x_2)}{(\\sqrt{2} b)}) \\right)
    /// \\end{split}
    /// $$ 
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
    
    fn sample<R: Rng>(&self, rng: &mut R, x: &<Int as Domain>::TypeDom, bandwidth: f64, dom: &Int) -> f64 {
        todo!()
    }
}

impl KernelFunc<Nat> for GaussianKernel
{
    /// Computes the Gaussian kernel value between two integer points, taking into account the bounds of the domain.
    /// The kernel value is computed as
    /// $$
    /// \\begin{split}
    ///     K(x_1, x_2) &= \\int_{x_1 - \frac{1}{2}}^{x_1 + \frac{1}{2}} K(x,x_2\\,|\\,b)dx \\\\
    ///                 &= \\frac{1}{2}\\left(\\text{erf}(\\frac{(\\frac{1}{2} - x_2 + x_1)}{(\\sqrt{2} b)}) -\\text{erf}(\\frac{(x_1 - \\frac{1}{2} - x_2)}{(\\sqrt{2} b)}) \\right)
    /// \\end{split}
    /// $$ 
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
    
    fn sample<R: Rng>(&self, rng: &mut R, x: &<Nat as Domain>::TypeDom, bandwidth: f64, dom: &Nat) -> f64 {
        todo!()
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
    
    fn sample<R: Rng>(&self, rng: &mut R, x: &<C as Domain>::TypeDom, bandwidth: f64, dom: &C) -> f64 {
        todo!()
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
            (MixedTypeDom::Real(r1), MixedTypeDom::Real(r2), Mixed::Real(d)) => {
                let kernel = GaussianKernel;
                kernel.compute(r1, r2, bandwidth, d)
            },
            (MixedTypeDom::Unit(u1), MixedTypeDom::Unit(u2), Mixed::Unit(d)) => {
                let kernel = GaussianKernel;
                kernel.compute(u1, u2, bandwidth, d)
            },
            (MixedTypeDom::Int(i1), MixedTypeDom::Int(i2), Mixed::Int(d)) => {
                let kernel = GaussianKernel;
                kernel.compute(i1, i2, bandwidth, d)
            },
            (MixedTypeDom::Nat(n1), MixedTypeDom::Nat(n2), Mixed::Nat(d)) => {
                let kernel = GaussianKernel;
                kernel.compute(n1, n2, bandwidth, d)
            },
            // Categorical
            (MixedTypeDom::Cat(c1), MixedTypeDom::Cat(c2), Mixed::Cat(d)) => {
                let kernel = AitchisonAitkenKernel;
                kernel.compute(c1, c2, bandwidth, d)
            },
            (MixedTypeDom::GridInt(i1), MixedTypeDom::GridInt(i2), Mixed::GridInt(d)) => {
                let kernel = AitchisonAitkenKernel;
                kernel.compute(i1, i2, bandwidth, d)
            },
            (MixedTypeDom::GridNat(n1), MixedTypeDom::GridNat(n2), Mixed::GridNat(d)) => {
                let kernel = AitchisonAitkenKernel;
                kernel.compute(n1, n2, bandwidth, d)
            },
            (MixedTypeDom::GridReal(r1), MixedTypeDom::GridReal(r2), Mixed::GridReal(d)) => {
                let kernel = AitchisonAitkenKernel;
                kernel.compute(r1, r2, bandwidth, d)
            },
            _ => panic!("Mismatched types in Mixed kernel computation"),
        }
    }
    
    fn sample<R: Rng>(&self, rng: &mut R, x: &<Mixed as Domain>::TypeDom, bandwidth: f64, dom: &Mixed) -> f64 {
        todo!()
    }
}

/// Trait for kernel functions used in the TPE algorithm. 
/// The kernel function computes the similarity between a given solution and the points in the archive, which is used to estimate the density of good and bad solutions.
/// 
/// # See also
/// - [`Univariate`] for the univariate kernel, which assumes independence between dimensions and computes the product of the kernel values for each dimension.
/// - [`Multivariate`] for the multivariate kernel, which models the joint distribution of all dimensions.
pub trait Kernel<D, Sp, S, SolId, SInfo>
where
    Self: Sized + Serialize + for<'a> Deserialize<'a>,
    D: Domain,
    Sp: Searchspace<S, SolId, SInfo, Opt=D>,
    S: Solution<SolId, LinkOpt<Sp>, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(&self, s1: &S::Raw, archive: &[S::Raw], searchspace: &Sp) -> f64;
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
        (MixedTypeDom::Cat(c1), MixedTypeDom::Cat(c2), Mixed::Cat(dom)) => {
            AitchisonAitkenKernel.compute(c1, c2, bandwidth, dom)
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

/// The univariate kernel computes the product of the kernel values for each dimension of the solution, assuming independence between dimensions.
/// For a solution of dimension $D$, the kernel value is computed as:
/// $$
/// K(\\mathbb{s}, \\{\\mathbb{s}\\}_{n=1}^N) = \\prod_{d=1}^{D} \\frac{1}{N} \\sum_{n=1}^N K_d(\mathbb{s}_{d}, \\mathbb{s}_{n,d} \\,\\lvert\\, b_d)\\enspace\\text{,}
/// $$
/// where $K_d$ is the kernel function ([`KernelFunc`]) for the $d$-th dimension, and $b_d$ is the bandwidth parameter for that dimension.
/// The bandwidth is computed using the Optuna rule [`optuna_bw`] for numerical dimensions, and [`cat_bw`] for categorical dimensions.
#[derive(Serialize, Deserialize)]
pub struct Univariate;

impl<Scp, S, SolId, SInfo> Kernel<Mixed, Scp, S, SolId, SInfo> for Univariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Mixed> + HasVariables,
    S: Solution<SolId, Mixed, SInfo, Raw = Arc<[TypeDom<Mixed>]>>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(
        &self,
        s1: &Arc<[TypeDom<Mixed>]>,
        archive: &[Arc<[TypeDom<Mixed>]>],
        searchspace: &Scp
    ) -> f64
    {
        let dim = searchspace.size();
        let size = archive.len();

        let mut product = 1.0;
        for (d, x1) in s1.iter().enumerate() {
            let mut sum = 0.0;
            let dom = searchspace.opt_at(d).unwrap();
            let b = compute_bw(size, dim, dom);
            for s2 in archive {
                let x2 = &s2[d];
                sum += compute_kernel(x1, x2, b, dom);
            }
            product *= sum / (size as f64);
        }
        product
    }
}

impl<Scp, S, SolId, SInfo> Kernel<Real, Scp, S, SolId, SInfo> for Univariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Real> + HasVariables,
    S: Solution<SolId, Real, SInfo, Raw = Arc<[TypeDom<Real>]>>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(
        &self,
        s1: &Arc<[TypeDom<Real>]>,
        archive: &[Arc<[TypeDom<Real>]>],
        searchspace: &Scp
    ) -> f64
    {
        let dim = searchspace.size();
        let size = archive.len();

        let mut product = 1.0;
        for (d, x1) in s1.iter().enumerate() {
            let mut sum = 0.0;
            let dom = searchspace.opt_at(d).unwrap();
            let b = optuna_bw(size, dim, dom);
            for s2 in archive {
                let x2 = &s2[d];
                sum += GaussianKernel.compute(x1, x2, b, dom);
            }
            product *= sum / (size as f64);
        }
        product
    }
}

impl<Scp, S, SolId, SInfo> Kernel<Int, Scp, S, SolId, SInfo> for Univariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Int> + HasVariables,
    S: Solution<SolId, Int, SInfo, Raw = Arc<[TypeDom<Int>]>>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(
        &self,
        s1: &Arc<[TypeDom<Int>]>,
        archive: &[Arc<[TypeDom<Int>]>],
        searchspace: &Scp
    ) -> f64
    {
        let dim = searchspace.size();
        let size = archive.len();

        let mut product = 1.0;
        for (d, x1) in s1.iter().enumerate() {
            let mut sum = 0.0;
            let dom = searchspace.opt_at(d).unwrap();
            let b = optuna_bw(size, dim, dom);
            for s2 in archive {
                let x2 = &s2[d];
                sum += GaussianKernel.compute(x1, x2, b, dom);
            }
            product *= sum / (size as f64);
        }
        product
    }
}

impl<Scp, S, SolId, SInfo> Kernel<Nat, Scp, S, SolId, SInfo> for Univariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Nat> + HasVariables,
    S: Solution<SolId, Nat, SInfo, Raw = Arc<[TypeDom<Nat>]>>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(
        &self,
        s1: &Arc<[TypeDom<Nat>]>,
        archive: &[Arc<[TypeDom<Nat>]>],
        searchspace: &Scp
    ) -> f64
    {
        let dim = searchspace.size();
        let size = archive.len();

        let mut product = 1.0;
        for (d, x1) in s1.iter().enumerate() {
            let mut sum = 0.0;
            let dom = searchspace.opt_at(d).unwrap();
            let b = optuna_bw(size, dim, dom);
            for s2 in archive {
                let x2 = &s2[d];
                sum += GaussianKernel.compute(x1, x2, b, dom);
            }
            product *= sum / (size as f64);
        }
        product
    }
}

impl<Scp, S, SolId, SInfo> Kernel<Unit, Scp, S, SolId, SInfo> for Univariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Unit> + HasVariables,
    S: Solution<SolId, Unit, SInfo, Raw = Arc<[TypeDom<Unit>]>>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(
        &self,
        s1: &Arc<[TypeDom<Unit>]>,
        archive: &[Arc<[TypeDom<Unit>]>],
        searchspace: &Scp
    ) -> f64
    {
        let dim = searchspace.size();
        let size = archive.len();

        let mut product = 1.0;
        for (d, x1) in s1.iter().enumerate() {
            let mut sum = 0.0;
            let dom = searchspace.opt_at(d).unwrap();
            let b = optuna_bw(size, dim, dom);
            for s2 in archive {
                let x2 = &s2[d];
                sum += GaussianKernel.compute(x1, x2, b, dom);
            }
            product *= sum / (size as f64);
        }
        product
    }
}

impl<T, Scp, S, SolId, SInfo> Kernel<GridDom<T>, Scp, S, SolId, SInfo> for Univariate 
where
    T: GridBounds,
    Scp: Searchspace<S, SolId, SInfo, Opt=GridDom<T>> + HasVariables,
    S: Solution<SolId, GridDom<T>, SInfo, Raw = Arc<[TypeDom<GridDom<T>>]>>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(
        &self,
        s1: &Arc<[TypeDom<GridDom<T>>]>,
        archive: &[Arc<[TypeDom<GridDom<T>>]>],
        searchspace: &Scp
    ) -> f64
    {
        let size = archive.len();

        let mut product = 1.0;
        for (d, x1) in s1.iter().enumerate() {
            let mut sum = 0.0;
            let dom = searchspace.opt_at(d).unwrap();
            let b = cat_bw(size, dom);
            for s2 in archive {
                let x2 = &s2[d];
                sum += AitchisonAitkenKernel.compute(x1, x2, b, dom);
            }
            product *= sum / (size as f64);
        }
        product
    }
}

/// The multivariate kernel computes the kernel value between two solutions by considering the joint distribution of all dimensions, without assuming independence.
/// The kernel value is computed as:
/// For a solution of dimension $D$, the kernel value is computed as:
/// $$
/// K(\\mathbb{s}, \\{\\mathbb{s}\\}_{n=1}^N) = \\frac{1}{N} \\sum_{n=1}^N \\prod_{d=1}^{D} K_d(\mathbb{s}_{d}, \\mathbb{s}_{n,d} \\,\\lvert\\, b_d)\\enspace\\text{,}
/// $$
/// where $K_d$ is the kernel function ([`KernelFunc`]) for the $d$-th dimension, and $b_d$ is the bandwidth parameter for that dimension.
/// The bandwidth is computed using the Optuna rule [`optuna_bw`] for numerical dimensions, and [`cat_bw`] for categorical dimensions.
#[derive(Serialize, Deserialize)]
pub struct Multivariate;

impl<Scp, S, SolId, SInfo> Kernel<Mixed, Scp, S, SolId, SInfo> for Multivariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Mixed> + HasVariables,
    S: Solution<SolId, Mixed, SInfo, Raw = Arc<[TypeDom<Mixed>]>>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(
        &self,
        s1: &Arc<[TypeDom<Mixed>]>,
        archive: &[Arc<[TypeDom<Mixed>]>],
        searchspace: &Scp
    ) -> f64
    {
        let dim = searchspace.size();
        let size = archive.len();

        let mut sum = 0.0;
        for s2 in archive {
            let mut product = 1.0;
            for (d, x1) in s1.iter().enumerate() {
                let dom = searchspace.opt_at(d).unwrap();
                let b = compute_bw(size, dim, dom);
                let x2 = &s2[d];
                product *= compute_kernel(x1, x2, b, dom);
            }
            sum += product;
        }
        sum / (size as f64)
    }
}

impl<Scp, S, SolId, SInfo> Kernel<Real, Scp, S, SolId, SInfo> for Multivariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Real> + HasVariables,
    S: Solution<SolId, Real, SInfo, Raw = Arc<[TypeDom<Real>]>>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(
        &self,
        s1: &Arc<[TypeDom<Real>]>,
        archive: &[Arc<[TypeDom<Real>]>],
        searchspace: &Scp
    ) -> f64
    {
        let dim = searchspace.size();
        let size = archive.len();

        let mut sum = 0.0;
        for s2 in archive {
            let mut product = 1.0;
            for (d, x1) in s1.iter().enumerate() {
                let dom = searchspace.opt_at(d).unwrap();
                let b = optuna_bw(size, dim, dom);
                let x2 = &s2[d];
                product *= GaussianKernel.compute(x1, x2, b, dom);
            }
            sum += product;
        }
        sum / (size as f64)
    }
}

impl<Scp, S, SolId, SInfo> Kernel<Int, Scp, S, SolId, SInfo> for Multivariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Int> + HasVariables,
    S: Solution<SolId, Int, SInfo, Raw = Arc<[TypeDom<Int>]>>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(
        &self,
        s1: &Arc<[TypeDom<Int>]>,
        archive: &[Arc<[TypeDom<Int>]>],
        searchspace: &Scp
    ) -> f64
    {
        let dim = searchspace.size();
        let size = archive.len();

        let mut sum = 0.0;
        for s2 in archive {
            let mut product = 1.0;
            for (d, x1) in s1.iter().enumerate() {
                let dom = searchspace.opt_at(d).unwrap();
                let b = optuna_bw(size, dim, dom);
                let x2 = &s2[d];
                product *= GaussianKernel.compute(x1, x2, b, dom);
            }
            sum += product;
        }
        sum / (size as f64)
    }
}

impl<Scp, S, SolId, SInfo> Kernel<Nat, Scp, S, SolId, SInfo> for Multivariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Nat> + HasVariables,
    S: Solution<SolId, Nat, SInfo, Raw = Arc<[TypeDom<Nat>]>>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(
        &self,
        s1: &Arc<[TypeDom<Nat>]>,
        archive: &[Arc<[TypeDom<Nat>]>],
        searchspace: &Scp
    ) -> f64
    {
        let dim = searchspace.size();
        let size = archive.len();

        let mut sum = 0.0;
        for s2 in archive {
            let mut product = 1.0;
            for (d, x1) in s1.iter().enumerate() {
                let dom = searchspace.opt_at(d).unwrap();
                let b = optuna_bw(size, dim, dom);
                let x2 = &s2[d];
                product *= GaussianKernel.compute(x1, x2, b, dom);
            }
            sum += product;
        }
        sum / (size as f64)
    }
}

impl<Scp, S, SolId, SInfo> Kernel<Unit, Scp, S, SolId, SInfo> for Multivariate 
where
    Scp: Searchspace<S, SolId, SInfo, Opt=Unit> + HasVariables,
    S: Solution<SolId, Unit, SInfo, Raw = Arc<[TypeDom<Unit>]>>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(
        &self,
        s1: &Arc<[TypeDom<Unit>]>,
        archive: &[Arc<[TypeDom<Unit>]>],
        searchspace: &Scp
    ) -> f64
    {
        let dim = searchspace.size();
        let size = archive.len();

        let mut sum = 0.0;
        for s2 in archive {
            let mut product = 1.0;
            for (d, x1) in s1.iter().enumerate() {
                let dom = searchspace.opt_at(d).unwrap();
                let b = optuna_bw(size, dim, dom);
                let x2 = &s2[d];
                product *= GaussianKernel.compute(x1, x2, b, dom);
            }
            sum += product;
        }
        sum / (size as f64)
    }
}

impl<T, Scp, S, SolId, SInfo> Kernel<GridDom<T>, Scp, S, SolId, SInfo> for Multivariate 
where
    T: GridBounds,
    Scp: Searchspace<S, SolId, SInfo, Opt=GridDom<T>> + HasVariables,
    S: Solution<SolId, GridDom<T>, SInfo, Raw = Arc<[TypeDom<GridDom<T>>]>>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn compute(
        &self,
        s1: &Arc<[TypeDom<GridDom<T>>]>,
        archive: &[Arc<[TypeDom<GridDom<T>>]>],
        searchspace: &Scp
    ) -> f64
    {
        let size = archive.len();

        let mut sum = 0.0;
        for s2 in archive {
            let mut product = 1.0;
            for (d, x1) in s1.iter().enumerate() {
                let dom = searchspace.opt_at(d).unwrap();
                let b = cat_bw(size, dom);
                let x2 = &s2[d];
                product *= AitchisonAitkenKernel.compute(x1, x2, b, dom);
            }
            sum += product;
        }
        sum / (size as f64)
    }
}