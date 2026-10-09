//! [Tree-structured Parzen Estimator (TPE)](https://arxiv.org/pdf/2304.11127) algorithms for black-box optimization.
//!
//! # Overview
//!
//! The objective of TPE is to generate on-demand new
//! [`Solution`](tantale_core::Solution)s to evaluate from the history of
//! previously evaluated configurations.
//!
//! This implementation is fully modular and supports both single-objective and
//! multi-objective optimization through interchangeable components:
//! - A [`Splitter`] partitions observations into *good* and *bad* sets.
//! - A [`Kernel`] builds density estimators for each parameter.
//! - A [`Bandwidth`] computes the bandwidth for the kernel density estimators.
//! - A [`Weighter`] assigns importance weights to observations.
//!
//! When enough observations are available, the optimizer fits two density
//! estimators, `l(x)` and `g(x)`, from the good and bad observations
//! respectively. Candidate configurations are then sampled from `l(x)` and
//! ranked according to the acquisition ratio `l(x) / g(x)`.
//!
//! If insufficient observations are available, the optimizer falls back to
//! random sampling.
//!
//! Single-objective and multi-objective variants are obtained by choosing
//! different [`Splitter`] implementations. In particular, multi-objective
//! splitters may rely on Pareto dominance, hypervolume contributions, or
//! other ranking criteria.
//!
//! # References
//!
//! - Watanabe, [*Tree-Structured Parzen Estimator: Understanding Its Algorithm Components and Their Roles for Better Empirical Performance*](https://arxiv.org/pdf/2304.11127).
//! - Ozaki et al., [*Multiobjective Tree-Structured Parzen Estimator*](https://www.jair.org/index.php/jair/article/view/13188/26784)
//!
use crate::bayesian::{bandwidth::Bandwidth, error::SplitError, kernel::Kernel, splitter::Splitter, weighter::Weighter};
use tantale_core::{
    HasFidelity, HasState, HasX, Orderable, OrderedArchive, Problem, Sampler, Searchspace, SolInfo, SolutionShape, State, Xy,
    aliases::{Accum, CodElem, CompShape, Opt, SOpt, Shape, XOpt}, problem::{FidelityMode, PbMode, SimpleMode}, utils::record::{Recordable, Schema},
};

use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tantale_macros::{basic_bounds, fidelity_bounds};
use core::panic;
use std::{cell::RefCell, marker::PhantomData, sync::Arc, vec};
use log::warn;

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(rand::make_rng());
}

/// A helper macro to shortcut the type signature of [`Tpe`] when loading.
#[macro_export]
macro_rules! tpe {
    ($kernel : ident, $bandwidth: ident, $weighter : ident, $splitter : ident) => {
        Tpe<$kernel, $bandwidth, $weighter, $splitter, _, _>
    };
}

/// Computes the acquisition function value based on the best and worse KDE-PDF values.
pub fn acquisition(good_pdf: f64, bad_pdf: f64) -> f64 {
    // if bad_pdf is 0, we want to return a very high acquisition value, to encourage exploration
    if bad_pdf == 0.0 {
        return f64::INFINITY;
    }
    if good_pdf == 0.0 {
        return 0.0;
    }
    good_pdf / bad_pdf
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct TpeSInfo {
    pub acquisition: f64,
    pub good_bdf: f64,
    pub bad_pdf: f64,
}

impl TpeSInfo {
    pub fn new(acquisition: f64, good_bdf: f64, bad_pdf: f64) -> Arc<Self> {
        Arc::new(TpeSInfo {
            acquisition,
            good_bdf,
            bad_pdf,
        })
    }
}

impl SolInfo for TpeSInfo {}

impl Schema<()> for TpeSInfo {
    type Parts = ();
    fn columns(_context: &(), _part: Self::Parts) -> Vec<String> {
        vec![
            "acquisition".to_string(),
            "good_bdf".to_string(),
            "bad_pdf".to_string(),
        ]
    }
}

impl Recordable<()> for TpeSInfo {
    type Parts = ();
    fn record(&self, _context: &(), _part: Self::Parts) -> Vec<String> {
        vec![
            self.acquisition.to_string(),
            self.good_bdf.to_string(),
            self.bad_pdf.to_string(),
        ]
    }
}

type JointFidelityArchive<Raw, CodElem> = Vec<(f64, OrderedArchive<Xy<Raw, CodElem>>)>;

/// [Tree-structured Parzen Estimator (TPE)](https://arxiv.org/pdf/2304.11127) optimizer for black-box optimization.
///
/// A [`SingleOptimizer`] implementing the TPE
/// family of algorithms for sequential model-based optimization.
///
/// Depending on the chosen [`Splitter`], the optimizer can be used for both
/// single-objective and multi-objective optimization.
/// * Single-objective optimization* is achieved by using a [`LinearSplit`](crate::bayesian::splitter::LinearSplit) or [`SqrtSplit`](crate::bayesian::splitter::SqrtSplit)
///   splitter, which partitions the observations into good and bad sets based on a single objective value.
/// * Multi-objective optimization* is achieved by using a [`MOSplit`](crate::bayesian::splitter::MOSplit)
///   splitter, which partitions the observations into good and bad sets based on Pareto dominance or hypervolume contributions.
///
/// # Overview
///
/// [`Tpe`] generates candidates on demand by fitting density estimators to the
/// history of evaluated configurations.
///
/// The optimizer is composed of three interchangeable components:
/// - A [`Splitter`] partitions the observations into *good* and *bad* sets.
///   In the multi-objective setting, this partition may rely on Pareto
///   dominance, hypervolume contributions, or other criteria.
/// - A [`Kernel`] builds density estimators for each parameter from the good
///   and bad observations.
/// - A [`Bandwidth`] computes the bandwidth for the kernel density estimators.
/// - A [`Weighter`] assigns importance weights to observations when fitting
///   the densities.
///
/// When a worker requests a new candidate, the optimizer:
/// 1. Splits the observations into good and bad sets.
/// 2. Fits two density models, `l(x)` and `g(x)`, corresponding to the good
///    and bad observations respectively.
/// 3. Samples candidate configurations from `l(x)`.
/// 4. Selects the candidate maximizing the acquisition ratio `l(x) / g(x)`.
///
/// If not enough observations are available, the optimizer falls back to
/// random sampling.
///
/// # Workflow
///
/// ```text
///  Worker requests solution
///           |
///           v
///  +---------------------------+
///  | Enough observations ?     |
///  +---------------------------+
///      Yes /       \ No
///         /         \
///        v           v
/// +----------------+   +----------------+
/// | Split history  |   | Random sample  |
/// | into good/bad  |   | configuration  |
/// +----------------+   +----------------+
///         |
///         v
/// +----------------+
/// | Fit kernels    |
/// | l(x) and g(x)  |
/// +----------------+
///         |
///         v
/// +----------------+
/// | Sample from    |
/// | l(x)           |
/// +----------------+
///         |
///         v
/// +----------------+
/// | Select x       |
/// | maximizing     |
/// | l(x) / g(x)    |
/// +----------------+
///         |
///         v
///   Return candidate
/// ```
///
/// # Internal State
///
/// - [`TpeState`]: Checkpointable state including:
///   * `n_init`: Number of initial random samples before fitting the model.
///   * `n_sample`: Number of candidate samples to draw from the good density `l(x)` when optimizing the acquisition function.
///   * `kernel`: The kernel used for density estimation.
///   * `weighter`: The weighter used for assigning importance weights to observations.
///   * `splitter`: The splitter used for partitioning observations into good and bad sets.
///   * `point_archive`: A collection of archived points.
///   * `current_fidelity`: The current fidelity level if applicable. Used for multi-fidelity optimization.
///   * `current_archive`: The current archive of solutions if applicable. Used for multi-fidelity optimization.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Wght: Serialize, Splt: Serialize, Bw: Serialize, Kern::Context: Serialize, Mode: Serialize",
    deserialize = "Wght: for<'a> Deserialize<'a>, Splt: for<'a> Deserialize<'a>, Bw: for<'a> Deserialize<'a>, Kern::Context: for<'a> Deserialize<'a>, Mode: for<'a> Deserialize<'a>",
))]
pub struct Tpe<Kern, Bw, Wght, Splt, P, Mode>
where
    Kern: Kernel<P, Opt<P>>,
    Bw: Bandwidth<P, Opt<P>>,
    Wght: Weighter<Xy<XOpt<P>, CodElem<P>>>,
    Splt: Splitter<Xy<XOpt<P>, CodElem<P>>>,
    P: Problem,
    Mode: PbMode,
    CodElem<P>: Orderable
{
    pub n_init: usize,
    pub n_sample: usize,
    pub kernel: Kern,
    pub bandwidth: Bw,
    pub weighter: Wght,
    pub splitter: Splt,
    pub point_archive: JointFidelityArchive<<SOpt<P> as HasX>::X, CodElem<P>>,
    pub current_fidelity: f64,
    pub current_archive: usize,
    _mode: PhantomData<Mode>,
}

impl<Kern, Bw, Wght, Splt, P, Mode> Tpe<Kern, Bw, Wght, Splt, P, Mode>
where
    Kern: Kernel<P, Opt<P>>,
    Bw: Bandwidth<P, Opt<P>>,
    Wght: Weighter<Xy<XOpt<P>, CodElem<P>>>,
    Splt: Splitter<Xy<XOpt<P>, CodElem<P>>>,
    P: Problem,
    Mode: PbMode,
    CodElem<P>: Orderable,
{
    /// Creates a new TPE optimizer with the specified components and parameters.
    ///
    /// # Parameters
    /// - `n_init`: The number of initial random samples to generate before fitting the model. Must be greater than 0.
    /// - `n_sample`: The number of candidate samples to draw from the good density `l(x)` when optimizing the acquisition function. Must be greater than 0.
    /// - `kernel`: The kernel used for density estimation. For example, [`Univariate`](crate::bayesian::Univariate) or [`Multivariate`](crate::bayesian::Multivariate).
    /// - `weighter`: The weighter used for assigning importance weights to observations. For example, [`UniformWeighter`](crate::bayesian::weighter::UniformWeighter).
    /// - `splitter`: The splitter used for partitioning observations into good and bad sets. For example, [`LinearSplit`](crate::bayesian::splitter::LinearSplit) or [`MOSplit`](crate::bayesian::splitter::MOSplit).
    pub fn new(
        n_init: usize,
        n_sample: usize,
        kernel: (Kern, Bw),
        weighter: Wght,
        splitter: Splt,
    ) -> Self {
        assert!(n_init > 0, "n_init must be greater than 0");
        assert!(n_sample > 0, "n_sample must be greater than 0");
        Tpe {
            n_init,
            n_sample,
            kernel: kernel.0,
            bandwidth: kernel.1,
            weighter,
            splitter,
            point_archive: vec![],
            current_archive: 0,
            current_fidelity: -f64::INFINITY,
            _mode: PhantomData,
        }
    }

    fn with_rng<F, A>(&self, f: F) -> A
    where
        F: FnOnce(&mut StdRng) -> A,
    {
        THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
    }
}

impl<Kern, Bw, Wght, Splt, P, Mode> State for Tpe<Kern, Bw, Wght, Splt, P, Mode>
where
    Kern: Kernel<P, Opt<P>>,
    Bw: Bandwidth<P, Opt<P>>,
    Wght: Weighter<Xy<XOpt<P>, CodElem<P>>>,
    Splt: Splitter<Xy<XOpt<P>, CodElem<P>>>,
    P: Problem,
    Mode: PbMode,
    CodElem<P>: Orderable,
{ }

impl<Kern, Bw, Wght, Splt, P, Mode> HasState for Tpe<Kern, Bw, Wght, Splt, P, Mode>
where
    Kern: Kernel<P, Opt<P>>,
    Bw: Bandwidth<P, Opt<P>>,
    Wght: Weighter<Xy<XOpt<P>, CodElem<P>>>,
    Splt: Splitter<Xy<XOpt<P>, CodElem<P>>>,
    P: Problem,
    Mode: PbMode,
    CodElem<P>: Orderable,
{
    type State = Self;

    fn get_mut_state(&mut self) -> &mut Self::State {
        self
    }

    fn get_state(&self) -> &Self::State {
        self
    }

    fn from_state(state: Self::State) -> Self {
        state
    }
}

#[basic_bounds(P, Opt, TpeSInfo)]
impl<Kern, Bw, Wght, Splt, P, Opt> Sampler<P, Opt> for Tpe<Kern, Bw, Wght, Splt, P, SimpleMode>
where
    Kern: Kernel<P, Opt>,
    Bw: Bandwidth<P, Opt>,
    Wght: Weighter<Xy<XOpt<P>, CodElem<P>>>,
    Splt: Splitter<Xy<XOpt<P>, CodElem<P>>>,
    CodElem<P>: Orderable,
{
    type Info = TpeSInfo;

    fn sample(&mut self,scp: &<P as Problem>::Scp,_acc: &Accum<P>) -> Option<Shape<P>> {
        let n_init = self.n_init;
        if self.point_archive[0].1.size() < n_init {
            let info = TpeSInfo::new(f64::NAN, f64::NAN, f64::NAN);
            Some(self.with_rng(|rng| scp.sample_pair(rng, info)))
        } else {
            // Split the archive into good and bad, and compute the weights
            let split = self.splitter.split(&self.point_archive[0].1);
            let (good, bad) = match split {
                Ok((good, bad)) => (good, bad),
                Err(err) => {
                    match err {
                        SplitError::NotEnoughPoints(msg) => {
                            warn!("Not enough points to split the archive: {msg}. Falling back to random sampling.");
                            let info = TpeSInfo::new(f64::NAN, f64::NAN, f64::NAN);
                            return Some(self.with_rng(|rng| scp.sample_pair(rng, info)));
                        },
                        _ => panic!("Unexpected error: {err:?}"),
                    }
                }
            };
            
            // Now bad and good set should not be empty
            let weights = self.weighter.weight(&good, &bad).unwrap(); // Weights for the good and bad points
            let good_bw =  self.bandwidth.compute(good.as_slice(), scp).unwrap();
            let good_ctx = Kern::get_context(good.as_slice(), scp, &good_bw).unwrap();
            let bad_bw =  self.bandwidth.compute(bad.as_slice(), scp).unwrap();
            let bad_ctx = Kern::get_context(bad.as_slice(), scp, &bad_bw).unwrap();

            let (s, acq, gpdf, bpdf) = (0..self.n_sample)
                .into_par_iter()
                .map(|_| {
                    let s = self.with_rng(|rng| {
                        self
                            .kernel
                            .sample(rng, &good, &good_ctx, scp)
                    });
                    let kernel = &self.kernel;
                    let good_pdf =
                        kernel.compute(&s, &good, &good_ctx, &weights.good, scp);
                    let bad_pdf = kernel.compute(&s, &bad, &bad_ctx, &weights.bad, scp);
                    let acq = acquisition(good_pdf, bad_pdf);
                    (s, acq, good_pdf, bad_pdf)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();

            let info = TpeSInfo::new(acq, gpdf, bpdf);
            Some(scp.new_opt(s, info))
        }
    }

    fn update(&mut self,x: &CompShape<P>,_scp: &<P as Problem>::Scp,_acc: &Accum<P>) {
        let xy = x.get_sopt().xy();
        self.point_archive[0].1.add(xy);
    }
}

#[fidelity_bounds(P, Opt, TpeSInfo)]
impl<Kern, Bw, Wght, Splt, P, Opt> Sampler<P, Opt> for Tpe<Kern, Bw, Wght, Splt, P, FidelityMode>
where
    Kern: Kernel<P, Opt>,
    Bw: Bandwidth<P, Opt>,
    Wght: Weighter<Xy<XOpt<P>, CodElem<P>>>,
    Splt: Splitter<Xy<XOpt<P>, CodElem<P>>>,
    CodElem<P>: Orderable,
{
    type Info = TpeSInfo;

    fn sample(&mut self,scp: &<P as Problem>::Scp,_acc: &Accum<P>) -> Option<Shape<P>> {
        if self.point_archive.is_empty() || self.point_archive[self.current_archive].1.size() < self.n_init {
            let info = TpeSInfo::new(f64::NAN, f64::NAN, f64::NAN);
            Some(self.with_rng(|rng| scp.sample_pair(rng, info)))
        } else {
            // Split the archive into good and bad, and compute the weights
            let split = self.splitter.split(&self.point_archive[self.current_archive].1);
            let (good, bad) = match split {
                Ok((good, bad)) => (good, bad),
                Err(err) => {
                    match err {
                        SplitError::NotEnoughPoints(msg) => {
                            warn!("Not enough points to split the archive: {msg}. Falling back to random sampling.");
                            let info = TpeSInfo::new(f64::NAN, f64::NAN, f64::NAN);
                            return Some(self.with_rng(|rng| scp.sample_pair(rng, info)));
                        },
                        _ => panic!("Unexpected error: {err:?}"),
                    }
                }
            };

            // Now bad and good set should not be empty
            let weights = self.weighter.weight(&good, &bad).unwrap(); // Weights for the good and bad points
            let good_bw =  self.bandwidth.compute(&good, scp).unwrap();
            let good_ctx = Kern::get_context(&good, scp, &good_bw).unwrap();
            let bad_bw =  self.bandwidth.compute(&bad, scp).unwrap();
            let bad_ctx = Kern::get_context(&bad, scp, &bad_bw).unwrap();

            let (s, acq, gpdf, bpdf) = (0..self.n_sample)
                .into_par_iter()
                .map(|_| {
                    let s = self.with_rng(|rng| {
                        self
                            .kernel
                            .sample(rng, &good, &good_ctx, scp)
                    });
                    let kernel = &self.kernel;
                    let good_pdf =
                        kernel.compute(&s, &good, &good_ctx, &weights.good, scp);
                    let bad_pdf = kernel.compute(&s, &bad, &bad_ctx, &weights.bad, scp);
                    let acq = acquisition(good_pdf, bad_pdf);
                    (s, acq, good_pdf, bad_pdf)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();

            let info = TpeSInfo::new(acq, gpdf, bpdf);
            Some(scp.new_opt(s, info))
        }
    }

    fn update(&mut self,x: &CompShape<P>,_scp: &<P as Problem>::Scp,_acc: &Accum<P>) {
        let fidelity = x.fidelity().0;

        if fidelity == self.current_fidelity {
            let xy = x.get_sopt().xy();
            self.point_archive[self.current_archive].1.add(xy);
        } else if fidelity > self.current_fidelity {
            let where_is = self.point_archive.iter().position(|f| f.0 == fidelity);
            // If the fidelity was already present in the archive.
            if let Some(idx) = where_is {
                let xy = x.get_sopt().xy();
                self.point_archive[idx].1.add(xy);

                // If the number of points at this fidelity exceeds the threshold, remove all previous fidelity points.
                if self.point_archive[idx].1.size() >= self.n_init {
                    self.point_archive.retain(|i| i.0 >= fidelity);
                    let i = self
                        .point_archive
                        .iter()
                        .position(|f| f.0 == fidelity)
                        .unwrap();
                    self.current_archive = i;
                    self.current_fidelity = fidelity;
                }
            } else {
                let xy = x.get_sopt().xy();
                self
                    .point_archive
                    .push((fidelity, OrderedArchive::new(xy)));
            }
        }
        // Else we ignore as an archive with higher accuracy has already reached the threshold
    }
}