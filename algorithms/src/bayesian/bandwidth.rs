use tantale_core::{Bool, Domain, GridDom, HasVariables, Id, Int, Mixed, MixedTypeDom, Nat, Outcome, Real, Searchspace, SolInfo, TypeCodom, Uncomputed, Unit, XToNdArray, Xy, domain::{CategoricalDomain, NumericalDomain, TypeDom, grid::GridBounds}};

use std::{mem::MaybeUninit, sync::Arc};
use num::{FromPrimitive, Num, cast::AsPrimitive};
use ndarray::{Array2, Axis, s};
use serde::{Deserialize, Serialize};

pub trait BandwidthType {
    fn get(&self, idx: usize, dim: usize) -> f64;
}

pub trait Bandwidth<Dom, Scp, S, SolId, SInfo, Out>
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Dom: Domain,
    Scp: Searchspace<S, SolId, SInfo, Opt = Dom>,
    S: Uncomputed<SolId, Dom, SInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType: BandwidthType;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType;
}

/// A struct representing the Optuna bandwidth method for Gaussian kernels.
/// 
/// Computes the bandwidth for a given point in the archive using Optuna's method:
/// $$
/// b = \\frac{U - L}{5} N^{-1/(d + 4)}\\enspace\\text{,}
/// $$
/// where $N$ is the number of points in the archive, $d$ is the dimensionality of the search space,
/// and $[L, U]$ are the bounds of the [`NumericalDomain`].
/// 
/// # Note
/// 
/// Categorical domains are handled differently, and the bandwidth is computed using a separate method:
/// $$
/// b = \\frac{N + 1}{N + C}\\enspace\\text{,}
/// $$
/// where $N$ is the number of points in the archive, and $C$ is the number of categories in the domain.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Optuna(bool);

impl Optuna {
    /// Creates a new instance of the Optuna bandwidth method.
    /// 
    /// # Arguments
    /// 
    /// * `magic_clip` - If `true`, then the bandwidth will be clipped using the magic clip method.
    /// 
    /// # Magic clip
    /// For a [`Bounded`](tantale_core::Bounded) ($[L, U]$) domain:
    /// $$
    /// b = \\max\\left(b, \\frac{U - L}{\\min(N, 100)}\\right)\\enspace\\text{,}
    /// $$
    /// where $b$ is the bandwidth computed by another methods, and $N$ is the number of points in the archive.
    pub fn new(magic_clip: bool) -> Self {
        Self(magic_clip)
    }
}

pub fn optuna_bw<D>(size: f64, neg_div_dim_p_four: f64, dom: &D, clip: bool) -> f64
where
    D: NumericalDomain,
    D::TypeDom: Num + AsPrimitive<f64> + FromPrimitive,
{
    let (low, up) = dom.get_bounds();
    let uml = (up.as_() - low.as_())/5.0;
    let bw = uml * size.powf(neg_div_dim_p_four);

    if clip {
        magic_clip(bw, size, uml)
    } else {
        bw
    }
}

pub fn cat_bw<D>(size: f64, dom: &D) -> f64
where
    D: CategoricalDomain,
{
    let c = dom.size() as f64;
    (size + 1.) / (size + c)
}

impl BandwidthType for Vec<f64> {
    fn get(&self, _idx: usize, dim: usize) -> f64 {
        self[dim]
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Real, Scp, S, SolId, SInfo, Out> for Optuna
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Real> + HasVariables,
    S: Uncomputed<SolId, Real, SInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Vec<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        let n_size = archive.len() as f64;
        
        let neg_div_dim_p_four = -1.0/(scp.size() + 4) as f64;

        scp.iter_opt().map(
            |dom| optuna_bw(n_size, neg_div_dim_p_four, dom, self.0)
        )
        .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Int, Scp, S, SolId, SInfo, Out> for Optuna
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Int> + HasVariables,
    S: Uncomputed<SolId, Int, SInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Vec<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        let n_size = archive.len() as f64;
        
        let neg_div_dim_p_four = -1.0/(scp.size() + 4) as f64;

        scp.iter_opt().map(
            |dom| optuna_bw(n_size, neg_div_dim_p_four, dom, self.0)
        )
        .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Nat, Scp, S, SolId, SInfo, Out> for Optuna
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Nat> + HasVariables,
    S: Uncomputed<SolId, Nat, SInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Vec<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        let n_size = archive.len() as f64;
        
        let neg_div_dim_p_four = -1.0/(scp.size() + 4) as f64;

        scp.iter_opt().map(
            |dom| optuna_bw(n_size, neg_div_dim_p_four, dom, self.0)
        )
        .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Unit, Scp, S, SolId, SInfo, Out> for Optuna
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Unit> + HasVariables,
    S: Uncomputed<SolId, Unit, SInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Vec<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        let n_size = archive.len() as f64;
        
        let neg_div_dim_p_four = -1.0/(scp.size() + 4) as f64;

        scp.iter_opt().map(
            |dom| optuna_bw(n_size, neg_div_dim_p_four, dom, self.0)
        )
        .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Mixed, Scp, S, SolId, SInfo, Out> for Optuna
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Mixed> + HasVariables,
    S: Uncomputed<SolId, Mixed, SInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Vec<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        let size = archive.len() as f64;
        let neg_div_dim_p_four = -1.0/(scp.size() + 4) as f64;

        scp.iter_opt().map(
            |dom|
             {
                match dom  {
                    Mixed::Real(dom) => optuna_bw(size, neg_div_dim_p_four, dom, self.0),
                    Mixed::Nat(dom) => optuna_bw(size, neg_div_dim_p_four, dom, self.0),
                    Mixed::Int(dom) => optuna_bw(size, neg_div_dim_p_four, dom, self.0),
                    Mixed::Unit(dom) => optuna_bw(size, neg_div_dim_p_four, dom, self.0),
                    Mixed::Bool(dom) => cat_bw(size, dom),
                    Mixed::Cat(dom) => cat_bw(size, dom),
                    Mixed::GridReal(dom) => cat_bw(size, dom),
                    Mixed::GridNat(dom) => cat_bw(size, dom),
                    Mixed::GridInt(dom) => cat_bw(size, dom),
                }
             }
        )
        .collect()
    }
}

fn hyperopt<'a, Raw, Y, I, D>(archive: &[&Xy<Raw, Y>], doms: I, consider_endpoint:bool, dim: usize, clip:bool) -> Array2<f64>
where
    I: Iterator<Item = &'a D>,
    D: NumericalDomain + 'a,
    D::TypeDom: Num + AsPrimitive<f64> + FromPrimitive + PartialOrd,
    Raw: XToNdArray<D>,
{
    let n_size = archive.len();
    let arr = archive.x_array();
    let mut res = Array2::uninit((n_size, dim));

    let mut indices = (0..archive.len()).collect::<Vec<_>>();
    for (colindex, dom) in doms.enumerate(){
        let (low, up) = dom.get_bounds();
        // make f64 comparable
        let col = arr.slice(s![.., colindex]);
        indices.sort_by(|&a, &b| col[a].partial_cmp(&col[b]).unwrap());
        indices.iter().enumerate().for_each(|(i, &idx)| {
            let prev : f64 = if i == 0 {
                if consider_endpoint {
                    low.as_()
                } else {
                    col[indices[i]].as_()
                }
            } else {
                col[indices[i - 1]].as_()
            };
            let next : f64 = if i == n_size - 1 {
                if consider_endpoint {
                    up.as_()
                } else {
                    col[indices[i]].as_()
                }
            } else {
                col[indices[i + 1]].as_()
            };
            let mut bw = (next - col[idx].as_()).max(col[idx].as_() - prev);
            if clip {
                bw = magic_clip(bw, n_size.as_(), (up - low).as_());
            }
            res[[idx, colindex]].write(bw);
        });
    }
    unsafe { res.assume_init() }
}

pub fn hyperopt_assign_col<D>(col: Vec<D::TypeDom>, colindex:usize, d: &D, res: &mut Array2<MaybeUninit<f64>>, consider_endpoint: bool, clip: bool)
where
    D: NumericalDomain,
    D::TypeDom: Num + AsPrimitive<f64> + FromPrimitive + PartialOrd,
{
    let n = col.len();
    let (low, up) = d.get_bounds();
    let mut indices: Vec<usize> = (0..n).collect();

    indices.sort_by(|&a, &b| col[a].partial_cmp(&col[b]).unwrap());

    for i in 0..n {
        let idx = indices[i];
        let xi = col[idx];

        let prev = if i == 0 {
            if consider_endpoint {
                low
            } else {
                col[indices[i]]
            }
        } else {
            col[indices[i - 1]]
        };
        let next = if i == n - 1 {
            if consider_endpoint {
                up
            } else {
                col[indices[i]]
            }
        } else {
            col[indices[i + 1]]
        };
        let mut bw = (next - xi).as_().max((xi - prev).as_());
        if clip {
            bw = magic_clip(bw, n.as_(), (up - low).as_());
        }
        res[[idx, colindex]].write(bw);
    }
}

fn hyperopt_mixed<
    'a,
    Y,
    DomIter,
>(
    archive: &[&Xy<Arc<[MixedTypeDom]>, Y>],
    doms: DomIter,
    consider_endpoint: bool,
    dim: usize,
    clip:bool,
) -> Array2<f64>
where
    DomIter: IntoIterator<Item = &'a Mixed>,
{
    let n = archive.len();
    let mut res = Array2::uninit((n, dim));

    for (colindex, dom) in doms.into_iter().enumerate() {

        match dom {

            // =========================
            // NUMERICAL CASES
            // =========================
            Mixed::Real(d) => {
                // Extract column
                let col: Vec<f64> = archive
                .iter()
                .map(|x| {
                    match x.x[colindex] {
                            MixedTypeDom::Real(e) => e,
                            _ => panic!("Unexpected non-numeric type in numeric column"),
                        }
                    })
                    .collect();
                hyperopt_assign_col(col, colindex, d, &mut res, consider_endpoint, clip);
            },
            Mixed::Int(d) => {
                // Extract column
                let col: Vec<i64> = archive
                    .iter()
                    .map(|x| {
                        match x.x[colindex] {
                            MixedTypeDom::Int(e) => e,
                            _ => panic!("Unexpected non-numeric type in numeric column"),
                        }
                    })
                    .collect();
               hyperopt_assign_col(col, colindex, d, &mut res, consider_endpoint, clip);
            },
            Mixed::Nat(d) => {
                // Extract column
                let col: Vec<u64> = archive
                    .iter()
                    .map(|x| {
                        match x.x[colindex] {
                            MixedTypeDom::Nat(e) => e,
                            _ => panic!("Unexpected non-numeric type in numeric column"),
                        }
                    })
                    .collect();
                hyperopt_assign_col(col, colindex, d, &mut res, consider_endpoint, clip);
            },
            Mixed::Unit(d) => {
                // Extract column
                let col: Vec<f64> = archive
                .iter()
                    .map(|x| {
                        match x.x[colindex] {
                            MixedTypeDom::Unit(e) => e,
                            _ => panic!("Unexpected non-numeric type in numeric column"),
                        }
                    })
                    .collect();
                hyperopt_assign_col(col, colindex, d, &mut res, consider_endpoint, clip);
            },
            // =========================
            // CATEGORICAL / COMPLEX TYPES
            // =========================
            Mixed::Cat(d) => {
                let bw = cat_bw(n.as_(), d);
                res.slice_mut(s![.., colindex]).fill(MaybeUninit::new(bw));
                
            },
            Mixed::Bool(d) => {
                let bw = cat_bw(n.as_(), d);
                res.slice_mut(s![.., colindex]).fill(MaybeUninit::new(bw));
                
            },
            Mixed::GridReal(d) => {
                let bw = cat_bw(n.as_(), d);
                res.slice_mut(s![.., colindex]).fill(MaybeUninit::new(bw));
                
            },
            Mixed::GridNat(d) => {
                let bw = cat_bw(n.as_(), d);
                res.slice_mut(s![.., colindex]).fill(MaybeUninit::new(bw));
                
            },
            Mixed::GridInt(d) =>{
                let bw = cat_bw(n.as_(), d);
                res.slice_mut(s![.., colindex]).fill(MaybeUninit::new(bw));
                
            }
        }
    }

    unsafe { res.assume_init() }
}

/// A struct representing the Hyperopt bandwidth method for Gaussian kernels.
/// 
/// Computes the bandwidth for a given point in the archive ($ \\{ x_0,\\, x_1,\\, \\dots\\,,\\, x_N,\\, x_{N+1} \\}$) using the Hyperopt method:
/// $$
/// b_n = \begin{cases}
///     x_n - x_{n-1} & \\text{if }x_n{n+1}\text{ not exist} \\\\
///     x_{n+1} - x_{n} & \\text{if }x_n{n-1}\text{ not exist} \\\\
///     \\max(x_{n+1} - x_n, x_n - x_{n-1}) & \\text{ otherwise} \\\\
/// \end{cases}
/// $$
/// where $x_n$ is the $n$-th point in the archive, and $x_{n-1}$ and $x_{n+1}$ are the previous and next points in the archive, respectively.
/// If [`consider_endpoint`](Hyperopt::consider_endpoint) is `true`, we consider $x_{N+1} = U$ and $x_0 = L$, with $[L, U]$ being the bounds of the [`NumericalDomain`].
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Hyperopt{
    /// If `true`, then consider the endpoints of the domain when computing the bandwidth.
    pub consider_endpoint: bool, 
    magic_clip: bool,
}

impl Hyperopt {
    /// Creates a new instance of the Hyperopt bandwidth method.
    /// 
    /// # Arguments
    /// 
    /// * `consider_endpoint` - If `true`, then consider the endpoints of the domain when computing the bandwidth.
    /// * `magic_clip` - If `true`, then use magic clipping when computing the bandwidth.
    pub fn new(consider_endpoint: bool, magic_clip: bool) -> Self {
        Self { consider_endpoint, magic_clip }
    }
}


impl BandwidthType for Array2<f64> {
    fn get(&self, idx: usize, dim: usize) -> f64 {
        self[[idx, dim]]
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Real, Scp, S, SolId, SInfo, Out> for Hyperopt
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Real> + HasVariables,
    S: Uncomputed<SolId, Real, SInfo, Raw = Arc<[TypeDom<Scp::Opt>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Array2<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        hyperopt(archive, scp.iter_opt(), self.consider_endpoint, scp.size(), self.magic_clip)
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Int, Scp, S, SolId, SInfo, Out> for Hyperopt
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Int> + HasVariables,
    S: Uncomputed<SolId, Int, SInfo, Raw = Arc<[TypeDom<Scp::Opt>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Array2<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        hyperopt(archive, scp.iter_opt(), self.consider_endpoint, scp.size(), self.magic_clip)
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Nat, Scp, S, SolId, SInfo, Out> for Hyperopt
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Nat> + HasVariables,
    S: Uncomputed<SolId, Nat, SInfo, Raw = Arc<[TypeDom<Scp::Opt>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Array2<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        hyperopt(archive, scp.iter_opt(), self.consider_endpoint, scp.size(), self.magic_clip)
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Unit, Scp, S, SolId, SInfo, Out> for Hyperopt
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Unit> + HasVariables,
    S: Uncomputed<SolId, Unit, SInfo, Raw = Arc<[TypeDom<Scp::Opt>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Array2<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        hyperopt(archive, scp.iter_opt(), self.consider_endpoint, scp.size(), self.magic_clip)
    }
}



impl<Scp, S, SolId, SInfo, Out> Bandwidth<Mixed, Scp, S, SolId, SInfo, Out> for Hyperopt
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Mixed> + HasVariables,
    S: Uncomputed<SolId, Mixed, SInfo, Raw = Arc<[TypeDom<Scp::Opt>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Array2<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        hyperopt_mixed(archive, scp.iter_opt(), self.consider_endpoint, scp.size(), self.magic_clip)
    }
}

/// Computes the bandwidth for categorical point in the archive using Optuna's method:
/// 
/// $$
/// b = \\frac{N + 1}{N + C}\\enspace\\text{,}
/// $$
/// where $N$ is the number of points in the archive, and $C$ is the number of categories in the domain.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CategoricalBw;

impl<T, Scp, S, SolId, SInfo, Out> Bandwidth<GridDom<T>, Scp, S, SolId, SInfo, Out> for Hyperopt
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    T: GridBounds,
    Scp: Searchspace<S, SolId, SInfo, Opt = GridDom<T>> + HasVariables,
    S: Uncomputed<SolId, GridDom<T>, SInfo, Raw = Arc<[TypeDom<Scp::Opt>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Vec<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        scp.iter_opt().map(
            |dom| cat_bw(archive.len().as_(), dom)
        )
        .collect()
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Bool, Scp, S, SolId, SInfo, Out> for Hyperopt
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Bool> + HasVariables,
    S: Uncomputed<SolId, Bool, SInfo, Raw = Arc<[TypeDom<Scp::Opt>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Vec<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        let n: f64 = archive.len().as_();
        vec![(n +1.) / (n + 2.) ; scp.size()]
    }
}

fn scott<'a, Raw, D, Y, I>(archive: &[&Xy<Raw, Y>], doms: I, clip: bool) -> Vec<f64>
where
    D: NumericalDomain + 'a,
    D::TypeDom: Num + AsPrimitive<f64>,
    Raw: XToNdArray<D>,
    I: Iterator<Item = &'a D>,
{
    let size= archive.len() as f64;
    let arr = archive.x_array_type::<f64>();
    let std = arr.std_axis(Axis(0), 0.0);
    let mut indices = (0..archive.len()).collect::<Vec<_>>();

    let mut res = Vec::new();
    for (colindex, dom) in doms.into_iter().enumerate(){
        let col = arr.slice(s![.., colindex]);
        indices.sort_by(|&a, &b| col[a].partial_cmp(&col[b]).unwrap());
        
        let q1_idx = indices[archive.len() / 4];
        let q3_idx = indices[(3 * archive.len() / 4).min(archive.len() - 1)];

        let q1 = col[q1_idx];
        let q3 = col[q3_idx];
        let iqr = q3 - q1;
        let mut bw = 1.059 * size.powf(-0.2) * std[colindex].min(iqr / 1.34).max(f64::EPSILON);
        if clip {
            let (low, up) = dom.get_bounds();
            bw = magic_clip(bw, size, (up - low).as_());
        }
        res.push(bw);
    }
    res
}

pub fn scott_col<D>(col: &[D::TypeDom], dom: &D, clip: bool) -> f64
where
    D: NumericalDomain,
    D::TypeDom: Num + AsPrimitive<f64>,

{
    let n = col.len();
    let size = n as f64;

    let mut values: Vec<f64> = col.iter().map(|x| x.as_()).collect();

    // Standard deviation (population)
    let mean = values.iter().sum::<f64>() / size;
    let std = (values
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / size)
        .sqrt();

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1 = values[n / 4];
    let q3 = values[(3 * n / 4).min(n - 1)];
    let iqr = q3 - q1;

    let mut bw =  (1.059 * size.powf(-0.2) * std.min(iqr / 1.34)).max(1e-12);
    if clip {
        let (low, up) = dom.get_bounds();
        bw = magic_clip(bw, size, (up - low).as_());
    }
    bw
}

fn scott_mixed<'a,Y,DomIter>(
    archive: &[&Xy<Arc<[MixedTypeDom]>, Y>],
    doms: DomIter,
    dim: usize,
    clip: bool
) -> Vec<f64>
where
    DomIter: IntoIterator<Item = &'a Mixed>,
{
    let n = archive.len();
    let mut res = Vec::with_capacity(dim);

    for (colindex, dom) in doms.into_iter().enumerate() {

        match dom {

            // =========================
            // NUMERICAL CASES
            // =========================
            Mixed::Real(dom) => {
                // Extract column
                let col: Vec<f64> = archive
                .iter()
                .map(|x| {
                    match x.x[colindex] {
                            MixedTypeDom::Real(e) => e,
                            _ => panic!("Unexpected non-numeric type in numeric column"),
                        }
                    })
                    .collect();
                res.push(scott_col(&col, dom, clip));
            },
            Mixed::Int(dom) => {
                // Extract column
                let col: Vec<i64> = archive
                    .iter()
                    .map(|x| {
                        match x.x[colindex] {
                            MixedTypeDom::Int(e) => e,
                            _ => panic!("Unexpected non-numeric type in numeric column"),
                        }
                    })
                    .collect();
               res.push(scott_col(&col, dom, clip));
            },
            Mixed::Nat(dom) => {
                // Extract column
                let col: Vec<u64> = archive
                    .iter()
                    .map(|x| {
                        match x.x[colindex] {
                            MixedTypeDom::Nat(e) => e,
                            _ => panic!("Unexpected non-numeric type in numeric column"),
                        }
                    })
                    .collect();
                res.push(scott_col(&col, dom, clip));
            },
            Mixed::Unit(dom) => {
                // Extract column
                let col: Vec<f64> = archive
                .iter()
                    .map(|x| {
                        match x.x[colindex] {
                            MixedTypeDom::Unit(e) => e,
                            _ => panic!("Unexpected non-numeric type in numeric column"),
                        }
                    })
                    .collect();
                res.push(scott_col(&col, dom, clip));
            },
            // =========================
            // CATEGORICAL / COMPLEX TYPES
            // =========================
            Mixed::Cat(d) => {
                res.push(cat_bw(n.as_(), d));
            },
            Mixed::Bool(d) => {
                res.push(cat_bw(n.as_(), d));
                
            },
            Mixed::GridReal(d) => {
                res.push(cat_bw(n.as_(), d));
                
            },
            Mixed::GridNat(d) => {
                res.push(cat_bw(n.as_(), d));
                
            },
            Mixed::GridInt(d) =>{
                res.push(cat_bw(n.as_(), d));
                
            }
        }
    }
    res
}

/// Computes the bandwidth for a given point in the archive using Scott's method:
/// $$
/// b = 1.059 N^{-1/5} \\min\\left(\\sigma, \\frac{IQR}{1.34}\\right)\\enspace\\text{,}
/// $$
/// where $N$ is the number of points in the archive, $\\sigma$ is the standard deviation of the points in the archive, and $IQR$ is the interquartile range of the points in the archive.
/// The interquartile range is computed as $IQR = Q_3 - Q_1$, where $Q_1$ and $Q_3$ are the first and third quartiles of the points in the archive, respectively
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Scott(bool);

impl Scott {
    /// Creates a new instance of the Scott bandwidth method.
    /// 
    /// # Arguments
    /// 
    /// * `magic_clip` - If `true`, then use magic clipping when computing the bandwidth.
    pub fn new(magic_clip: bool) -> Self {
        Self(magic_clip)
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Real, Scp, S, SolId, SInfo, Out> for Scott
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Real> + HasVariables,
    S: Uncomputed<SolId, Real, SInfo, Raw = Arc<[TypeDom<Scp::Opt>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Vec<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        scott::<S::Raw, Real, _, _>(archive, scp.iter_opt(), self.0)
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Int, Scp, S, SolId, SInfo, Out> for Scott
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Int> + HasVariables,
    S: Uncomputed<SolId, Int, SInfo, Raw = Arc<[TypeDom<Scp::Opt>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Vec<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        scott::<S::Raw, Int, _, _>(archive, scp.iter_opt(), self.0)
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Nat, Scp, S, SolId, SInfo, Out> for Scott
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Nat> + HasVariables,
    S: Uncomputed<SolId, Nat, SInfo, Raw = Arc<[TypeDom<Scp::Opt>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Vec<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        scott::<S::Raw, Nat, _, _>(archive, scp.iter_opt(), self.0)
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Unit, Scp, S, SolId, SInfo, Out> for Scott
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Unit> + HasVariables,
    S: Uncomputed<SolId, Unit, SInfo, Raw = Arc<[TypeDom<Scp::Opt>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Vec<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        scott::<S::Raw, Unit, _, _>(archive, scp.iter_opt(), self.0)
    }
}

impl<Scp, S, SolId, SInfo, Out> Bandwidth<Mixed, Scp, S, SolId, SInfo, Out> for Scott
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    Scp: Searchspace<S, SolId, SInfo, Opt = Mixed> + HasVariables,
    S: Uncomputed<SolId, Mixed, SInfo, Raw = Arc<[TypeDom<Scp::Opt>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    type BwType = Vec<f64>;
    fn compute(&mut self, archive: &[&Xy<S::Raw, TypeCodom<Out>>], scp: &Scp) -> Self::BwType{
        scott_mixed(archive, scp.iter_opt(), scp.size(), self.0)
    }
}

/// "Magic clipping" function.
/// For a [`Bounded`](tantale_core::Bounded) ($[L, U]$) domain:
/// $$
/// b = \\max\\left(b, \\frac{U - L}{\\min(N, 100)}\\right)\\enspace\\text{,}
/// $$
/// where $b$ is the bandwidth computed by another methods, and $N$ is the number of points in the archive.
pub fn magic_clip(bandwidth: f64, size: f64, range: f64) -> f64
{
    bandwidth.max(range /  size.min(100.))
}
