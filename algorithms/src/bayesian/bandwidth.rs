use tantale_core::domain::{CategoricalDomain, NumericalDomain};

use num::{Num, cast::AsPrimitive};

#[deprecated(note = "Experimental API - may change or be removed")]
/// Bandwidth methods for Gaussian kernels.
pub enum GaussianBandwidth {
    /// Hyperopt method to compute the bandwidth.
    /// See [`hyperopt_bw`] for more details.
    /// If `true`, then apply "magic clipping" to the bandwidth.
    Hyperopt(bool),
    /// Scott method to compute the bandwidth.
    /// See [`scott_bw`] for more details.
    /// If `true`, then apply "magic clipping" to the bandwidth.
    Scott(bool),
    /// BOHB method to compute the bandwidth.
    /// See [`optuna_bw`] for more details.
    /// If `true`, then apply "magic clipping" to the bandwidth.
    Bohb(bool),
    /// Optuna method to compute the bandwidth.
    /// See [`optuna_bw`] for more details.
    /// If `true`, then apply "magic clipping" to the bandwidth.
    Optuna(bool),
}

#[deprecated(note = "Experimental API - may change or be removed")]
/// Bandwidth methods for Aitchison-Aitken kernels.
pub enum AitchisonAitkenBandwidth {
    /// Categorical bandwidth method to compute the bandwidth.
    /// See [`cat_bw`] for more details.
    Optuna,
}

/// Computes the bandwidth for a given point in the archive ($ \\{ x_0,\\, x_1,\\, \\dots\\,,\\, x_N,\\, x_{N+1} \\}$) using the Hyperopt method:
/// $$
/// b_n = \\max(x_{n+1} - x_n, x_n - x_{n-1})\\enspace\\text{,}
/// $$
/// where $x_n$ is the $n$-th point in the archive, and $x_{n-1}$ and $x_{n+1}$ are the previous and next points in the archive, respectively.
/// We consider $x_{N+1} = U$ and $x_0 = L$, with $[L, U]$ being the bounds of the [`NumericalDomain`].
///
/// # Parameters
/// - `archive`: A vector of elements at index $d$ within a vector-like solution of size $d$.
///   The `archive` is assumed to be sorted in ascending order $x_1 \leq x_2 \leq \ldots \leq x_N$.
/// - `n`: The index of the point for which to compute the bandwidth, where $0 \leq n < N$.
/// - `dom`: The numerical domain within which the points in the archive are defined.
#[deprecated(note = "Experimental API - may change or be removed")]
pub fn hyperopt_bw<D>(archive: &[&D::TypeDom], n: usize, dom: &D) -> f64
where
    D: NumericalDomain,
    D::TypeDom: Num + AsPrimitive<f64>,
{
    let xn = archive[n].as_();
    let prev;
    let next;
    if archive.len() < 2 {
        if n == 0 {
            prev = dom.get_bounds().0.as_();
            next = dom.get_bounds().1.as_();
        } else {
            panic!("Cannot compute bandwidth with n > 0 and archive length < 2");
        }
    } else if n == 0 {
        prev = dom.get_bounds().0.as_();
        next = archive[n + 1].as_();
    } else if n == archive.len() - 1 {
        prev = archive[n - 1].as_();
        next = dom.get_bounds().1.as_();
    } else {
        prev = archive[n - 1].as_();
        next = archive[n + 1].as_();
    }
    (next - xn).max(xn - prev)
}

/// Computes the bandwidth for a given point in the archive using Scott's method:
/// $$
/// b = 1.059 N^{-1/5} \\min\\left(\\sigma, \\frac{IQR}{1.34}\\right)\\enspace\\text{,}
/// $$
/// where $N$ is the number of points in the archive, $\\sigma$ is the standard deviation of the points in the archive, and $IQR$ is the interquartile range of the points in the archive.
/// The interquartile range is computed as $IQR = Q_3 - Q_1$, where $Q_1$ and $Q_3$ are the first and third quartiles of the points in the archive, respectively.
#[deprecated(note = "Experimental API - may change or be removed")]
pub fn scott_bw<D>(archive: &mut [&D::TypeDom]) -> f64
where
    D: NumericalDomain,
    D::TypeDom: Num + AsPrimitive<f64> + Ord,
{
    archive.sort();
    let size = archive.len() as f64;

    let mean = archive.iter().map(|x| x.as_()).sum::<f64>() / size;
    let variance = archive
        .iter()
        .map(|x| (x.as_() - mean).powi(2))
        .sum::<f64>()
        / (size - 1.0);
    let std_err = variance.sqrt();

    let q1 = archive[archive.len() / 4].as_();
    let q3 = archive[(3 * archive.len() / 4).min(archive.len() - 1)].as_();
    let iqr = q3 - q1;

    1.059 * size.powf(-1.0 / 5.0) * std_err.min(iqr / 1.34)
}

/// Computes the bandwidth for a given point in the archive using Optuna's method:
/// $$
/// b = \\frac{U - L}{5} N^{-1/(d + 4)}\\enspace\\text{,}
/// $$
/// where $N$ is the number of points in the archive, $d$ is the dimensionality of the search space,
/// and $[L, U]$ are the bounds of the [`NumericalDomain`].
pub fn optuna_bw<D>(size: usize, dim: usize, dom: &D) -> f64
where
    D: NumericalDomain,
    D::TypeDom: Num + AsPrimitive<f64>,
{
    let size = size as f64;
    let (low, up) = dom.get_bounds();
    (up.as_() - low.as_()) / 5.0 * size.powf(-1.0 / ((dim + 4) as f64))
}

/// Computes the bandwidth for a given point in the archive for [`CategoricalDomain`] using the method:
/// $$
/// b = \\frac{C - 1}{N + C}\\enspace\\text{,}
/// $$
/// where $N$ is the number of points in the archive, and $C$ is the number of categories in the domain.
pub fn cat_bw<D>(size: usize, dom: &D) -> f64
where
    D: CategoricalDomain,
{
    let n = size as f64;
    let c = dom.size() as f64;
    (c - 1.) / (n + c)
}

/// "Magic clipping" function.
/// For a [`Bounded`](tantale_core::Bounded) ($[L, U]$) domain:
/// $$
/// b = \\max\\left(b, \\frac{U - L}{\\min(N, 100)}\\right)\\enspace\\text{,}
/// $$
/// where $b$ is the bandwidth computed by another methods, and $N$ is the number of points in the archive.
pub fn magic_clip<D>(bandwidth: f64, size: usize, dom: &D) -> f64
where
    D: NumericalDomain,
    D::TypeDom: Num + AsPrimitive<f64>,
{
    let (low, up) = dom.get_bounds();
    let range = up - low;
    let size = size.min(100) as f64;
    bandwidth.max(range.as_()/ size)
}
