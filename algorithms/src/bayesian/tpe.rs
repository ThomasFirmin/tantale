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
use crate::{
    bayesian::{kernel::Kernel, splitter::Splitter, weighter::Weighter},
    utils::{BCompAcc, BCompShape, FCompAcc, FCompShape, SimpleObjective, SimpleStepped},
};
use tantale_core::{
    BaseSol, CSVWritable, CompAcc, CompShape, FidOutcome, FidelitySol, FuncState, HasFidelity,
    HasStep, Id, IntoComputedShape, LinkOpt, OptState, Optimizer, Orderable, OrderedArchive,
    Outcome, SId, Sampler, Searchspace, SingleOptimizer, SingleSampler, SolInfo, SolutionShape,
    Step, StepSId, Stepped, Uncomputed, Xy,
    domain::{TypeDom, codomain::TypeCodom},
    solution::shape::RawOpt,
};

use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, sync::Arc};

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(rand::make_rng());
}

/// A helper macro to simplify the type signature of [`Tpe`].
/// For example, to load a TPE optimizer with `Univariate`, `UniformWeighter` and `LinearSplit`, instead of writing:
/// ```
/// let exp = load!(mono, Tpe<Univariate, UniformWeighter, LinearSplit, _, _, _, _, _>, Evaluated, (sp, cod), obj, (rec, check));
/// ```
/// you can write:
/// ```
/// let exp = load!(mono, tpe!(Univariate, UniformWeighter, LinearSplit), Evaluated, (sp, cod), obj, (rec, check));
/// ```
#[macro_export]
macro_rules! tpe {
    ($kernel : ident, $weighter : ident, $splitter : ident) => {
        Tpe<$kernel, $weighter, $splitter, _, _, _, _>
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

impl CSVWritable<(), ()> for TpeSInfo {
    fn header(_elem: &()) -> Vec<String> {
        vec![
            "acquisition".to_string(),
            "good_bdf".to_string(),
            "bad_pdf".to_string(),
        ]
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        vec![format!(
            "{},{},{}",
            self.acquisition, self.good_bdf, self.bad_pdf
        )]
    }
}

type JointFidelityArchive<Raw, CodElem> = Vec<(f64, OrderedArchive<Xy<Raw, CodElem>>)>;

/// Internal state of the [`Tpe`] optimizer.
///
/// This structure maintains all essential information needed to resume an optimization
/// across checkpoints. It encodes the core parameters of the algorithm.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Wght: Serialize, Splt: Serialize, Kern::KContext: Serialize, Kern::SContext: Serialize",
    deserialize = "Wght: for<'a> Deserialize<'a>, Splt: for<'a> Deserialize<'a>, Kern::KContext: for<'a> Deserialize<'a>, Kern::SContext: for<'a> Deserialize<'a>",
))]
pub struct TpeState<Kern, Wght, Splt, Scp, S, SolId, Out>
where
    S: Uncomputed<SolId, Scp::Opt, TpeSInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, TpeSInfo>,
    Scp: Searchspace<S, SolId, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo, Out>,
    Wght: Weighter<Xy<S::Raw, TypeCodom<Out>>>,
    Splt: Splitter<Xy<S::Raw, TypeCodom<Out>>>,
    SolId: Id,
    TypeCodom<Out>: Orderable,
    Out: Outcome,
{
    pub n_init: usize,
    pub n_sample: usize,
    pub kernel: Kern,
    pub weighter: Wght,
    pub splitter: Splt,
    pub point_archive: JointFidelityArchive<S::Raw, TypeCodom<Out>>,
    pub current_fidelity: f64,
    pub current_archive: usize,
}

impl<Kern, Wght, Splt, Scp, S, SolId, Out> OptState
    for TpeState<Kern, Wght, Splt, Scp, S, SolId, Out>
where
    S: Uncomputed<SolId, Scp::Opt, TpeSInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, TpeSInfo>,
    Scp: Searchspace<S, SolId, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo, Out>,
    Wght: Weighter<Xy<S::Raw, TypeCodom<Out>>>,
    Splt: Splitter<Xy<S::Raw, TypeCodom<Out>>>,
    SolId: Id,
    TypeCodom<Out>: Orderable,
    Out: Outcome,
{
}

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
pub struct Tpe<Kern, Wght, Splt, Scp, S, SolId, Out>(
    pub TpeState<Kern, Wght, Splt, Scp, S, SolId, Out>,
)
where
    Scp: Searchspace<S, SolId, TpeSInfo>,
    S: Uncomputed<SolId, Scp::Opt, TpeSInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo, Out>,
    Wght: Weighter<Xy<S::Raw, TypeCodom<Out>>>,
    Splt: Splitter<Xy<S::Raw, TypeCodom<Out>>>,
    SolId: Id,
    TypeCodom<Out>: Orderable,
    Out: Outcome;

impl<Kern, Wght, Splt, Scp, S, SolId, Out> Tpe<Kern, Wght, Splt, Scp, S, SolId, Out>
where
    Scp: Searchspace<S, SolId, TpeSInfo>,
    S: Uncomputed<SolId, Scp::Opt, TpeSInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo, Out>,
    Wght: Weighter<Xy<S::Raw, TypeCodom<Out>>>,
    Splt: Splitter<Xy<S::Raw, TypeCodom<Out>>>,
    SolId: Id,
    TypeCodom<Out>: Orderable,
    Out: Outcome,
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
        kernel: Kern,
        weighter: Wght,
        splitter: Splt,
    ) -> Self {
        assert!(n_init > 0, "n_init must be greater than 0");
        assert!(n_sample > 0, "n_sample must be greater than 0");
        Tpe(TpeState {
            n_init,
            n_sample,
            kernel,
            weighter,
            splitter,
            point_archive: vec![(0.0, OrderedArchive::default())],
            current_archive: 0,
            current_fidelity: 0.0,
        })
    }

    fn with_rng<F, A>(&self, f: F) -> A
    where
        F: FnOnce(&mut StdRng) -> A,
    {
        THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
    }
}

impl<Kern, Wght, Splt, Scp, Out>
    Optimizer<BaseSol<SId, Scp::Opt, TpeSInfo>, SId, Scp::Opt, Out, Scp>
    for Tpe<Kern, Wght, Splt, Scp, BaseSol<SId, Scp::Opt, TpeSInfo>, SId, Out>
where
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo>,
    Kern: Kernel<LinkOpt<Scp>, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, SId, TpeSInfo>, TypeCodom<Out>>>,
    Splt: Splitter<Xy<RawOpt<Scp::SolShape, SId, TpeSInfo>, TypeCodom<Out>>>,
    TypeCodom<Out>: Orderable,
    Out: Outcome,
{
    type State = TpeState<Kern, Wght, Splt, Scp, BaseSol<SId, Scp::Opt, TpeSInfo>, SId, Out>;
    type SInfo = TpeSInfo;

    fn get_state(&self) -> &Self::State {
        &self.0
    }

    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state)
    }
}

impl<Kern, Wght, Splt, Scp, Out>
    Optimizer<FidelitySol<StepSId, Scp::Opt, TpeSInfo>, StepSId, Scp::Opt, Out, Scp>
    for Tpe<Kern, Wght, Splt, Scp, FidelitySol<StepSId, Scp::Opt, TpeSInfo>, StepSId, Out>
where
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo>,
    Kern: Kernel<
            LinkOpt<Scp>,
            Scp,
            FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>,
            StepSId,
            TpeSInfo,
            Out,
        >,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, StepSId, TpeSInfo>, TypeCodom<Out>>>,
    Splt: Splitter<Xy<RawOpt<Scp::SolShape, StepSId, TpeSInfo>, TypeCodom<Out>>>,
    TypeCodom<Out>: Orderable,
    Out: Outcome,
{
    type State =
        TpeState<Kern, Wght, Splt, Scp, FidelitySol<StepSId, Scp::Opt, TpeSInfo>, StepSId, Out>;
    type SInfo = TpeSInfo;

    fn get_state(&self) -> &Self::State {
        &self.0
    }

    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state)
    }
}

// Implementation for Mixed-NoDomain searchspace
impl<Kern, Wght, Splt, Scp, Out>
    SingleOptimizer<
        BaseSol<SId, LinkOpt<Scp>, TpeSInfo>,
        SId,
        LinkOpt<Scp>,
        Out,
        Scp,
        SimpleObjective<Scp::SolShape, TpeSInfo, Out>,
    > for Tpe<Kern, Wght, Splt, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, Out>
where
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo> + Send + Sync,
    Kern: Kernel<LinkOpt<Scp>, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, Out>
        + Send
        + Sync,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, SId, TpeSInfo>, TypeCodom<Out>>> + Send + Sync,
    Splt: Splitter<Xy<RawOpt<Scp::SolShape, SId, TpeSInfo>, TypeCodom<Out>>> + Send + Sync,
    Out: Outcome,
    TypeCodom<Out>: Send + Sync + Orderable,
    TypeDom<Scp::Opt>: Send + Sync,
    Kern::KContext: Send + Sync,
    Kern::SContext: Send + Sync,
{
    fn step(
        &mut self,
        x: Option<BCompShape<Scp, Out, TpeSInfo>>,
        scp: &Scp,
        _acc: &BCompAcc<Scp, Out, TpeSInfo>,
    ) -> Scp::SolShape {
        if let Some(point) = x {
            let xy = point.get_sopt().xy();
            self.0.point_archive[0].1.add(xy);
        }

        let n_init = self.0.n_init;
        if self.0.point_archive[0].1.size() < n_init {
            let info = TpeSInfo::new(0.0, 0.0, 0.0);
            self.with_rng(|rng| scp.sample_pair(rng, info))
        } else {
            // Split the archive into good and bad, and compute the weights
            let (good, bad) = self.0.splitter.split(&self.0.point_archive[0].1);
            let weights = self.0.weighter.weight(&good, &bad); // Weights for the good and bad points
            let (good_sctx, good_kctx) = Kern::get_context(&good, scp);
            let (bad_sctx, bad_kctx) = Kern::get_context(&bad, scp);

            let (s, acq, gpdf, bpdf) = (0..self.0.n_sample)
                .into_par_iter()
                .map(|_| {
                    let s = self.with_rng(|rng| {
                        self.0
                            .kernel
                            .sample(rng, &good, &good_kctx, &good_sctx, scp)
                    });
                    let kernel = &self.0.kernel;
                    let good_pdf =
                        kernel.compute(&s, &good, &good_kctx, &good_sctx, &weights.good, scp);
                    let bad_pdf = kernel.compute(&s, &bad, &bad_kctx, &bad_sctx, &weights.bad, scp);
                    let acq = acquisition(good_pdf, bad_pdf);
                    (s, acq, good_pdf, bad_pdf)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();

            let info = TpeSInfo::new(acq, gpdf, bpdf);
            scp.new_opt(s, info)
        }
    }
}

impl<Kern, Wght, Splt, Scp, Out, FnState>
    SingleOptimizer<
        FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>,
        StepSId,
        LinkOpt<Scp>,
        Out,
        Scp,
        SimpleStepped<Scp::SolShape, TpeSInfo, Out, FnState>,
    > for Tpe<Kern, Wght, Splt, Scp, FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, Out>
where
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo> + Send + Sync,
    Kern: Kernel<
            LinkOpt<Scp>,
            Scp,
            FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>,
            StepSId,
            TpeSInfo,
            Out,
        > + Send
        + Sync,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, StepSId, TpeSInfo>, TypeCodom<Out>>> + Send + Sync,
    Splt: Splitter<Xy<RawOpt<Scp::SolShape, StepSId, TpeSInfo>, TypeCodom<Out>>> + Send + Sync,
    TypeCodom<Out>: Send + Sync + Orderable,
    TypeDom<Scp::Opt>: Send + Sync,
    Kern::KContext: Send + Sync,
    Kern::SContext: Send + Sync,
    Out: FidOutcome,
    FnState: FuncState,
    Scp::SolShape: HasStep + HasFidelity,
    FCompShape<Scp, Out, TpeSInfo>: HasStep + HasFidelity,
{
    fn step(
        &mut self,
        x: Option<FCompShape<Scp, Out, TpeSInfo>>,
        scp: &Scp,
        acc: &FCompAcc<Scp, Out, TpeSInfo>,
    ) -> Scp::SolShape {
        if let Some(point) = x {
            match point.step() {
                Step::Pending => {
                    unreachable!("A pending SolShape, should not be passed to RandomSearch step.")
                }
                Step::Partially(_) => IntoComputedShape::extract(point).0,
                Step::Evaluated => {
                    let xy = point.get_sopt().xy();
                    self.0.point_archive[0].1.add(xy);
                    <Tpe<
                        Kern,
                        Wght,
                        Splt,
                        Scp,
                        FidelitySol<StepSId, Scp::Opt, TpeSInfo>,
                        StepSId,
                        Out,
                    > as SingleOptimizer<
                        FidelitySol<StepSId, Scp::Opt, TpeSInfo>,
                        StepSId,
                        Scp::Opt,
                        Out,
                        Scp,
                        Stepped<Arc<[TypeDom<Scp::Obj>]>, Out, FnState>,
                    >>::step(self, None, scp, acc)
                }
                _ => <Tpe<
                    Kern,
                    Wght,
                    Splt,
                    Scp,
                    FidelitySol<StepSId, Scp::Opt, TpeSInfo>,
                    StepSId,
                    Out,
                > as SingleOptimizer<
                    FidelitySol<StepSId, Scp::Opt, TpeSInfo>,
                    StepSId,
                    Scp::Opt,
                    Out,
                    Scp,
                    Stepped<Arc<[TypeDom<Scp::Obj>]>, Out, FnState>,
                >>::step(self, None, scp, acc),
            }
        } else {
            let n_init = self.0.n_init;
            if self.0.point_archive[0].1.size() < n_init {
                let info = TpeSInfo::new(0.0, 0.0, 0.0);
                self.with_rng(|rng| scp.sample_pair(rng, info))
            } else {
                // Split the archive into good and bad, and compute the weights
                let (good, bad) = self.0.splitter.split(&self.0.point_archive[0].1);
                let weights = self.0.weighter.weight(&good, &bad); // Weights for the good and bad points
                let (good_sctx, good_kctx) = Kern::get_context(&good, scp);
                let (bad_sctx, bad_kctx) = Kern::get_context(&bad, scp);

                let (s, acq, gpdf, bpdf) = (0..self.0.n_sample)
                    .into_par_iter()
                    .map(|_| {
                        let s = self.with_rng(|rng| {
                            self.0
                                .kernel
                                .sample(rng, &good, &good_kctx, &good_sctx, scp)
                        });
                        let kernel = &self.0.kernel;
                        let good_pdf =
                            kernel.compute(&s, &good, &good_kctx, &good_sctx, &weights.good, scp);
                        let bad_pdf =
                            kernel.compute(&s, &bad, &bad_kctx, &bad_sctx, &weights.bad, scp);
                        let acq = acquisition(good_pdf, bad_pdf);
                        (s, acq, good_pdf, bad_pdf)
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap();

                let info = TpeSInfo::new(acq, gpdf, bpdf);
                scp.new_opt(s, info)
            }
        }
    }
}

impl<Kern, Wght, Splt, Scp, Out> Sampler<BaseSol<SId, Scp::Opt, TpeSInfo>, SId, Scp::Opt, Out, Scp>
    for Tpe<Kern, Wght, Splt, Scp, BaseSol<SId, Scp::Opt, TpeSInfo>, SId, Out>
where
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo>,
    Kern: Kernel<LinkOpt<Scp>, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, SId, TpeSInfo>, TypeCodom<Out>>>,
    Splt: Splitter<Xy<RawOpt<Scp::SolShape, SId, TpeSInfo>, TypeCodom<Out>>>,
    TypeCodom<Out>: Orderable,
    Out: Outcome,
{
}

impl<Kern, Wght, Splt, Scp, Out>
    Sampler<FidelitySol<StepSId, Scp::Opt, TpeSInfo>, StepSId, Scp::Opt, Out, Scp>
    for Tpe<Kern, Wght, Splt, Scp, FidelitySol<StepSId, Scp::Opt, TpeSInfo>, StepSId, Out>
where
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo>,
    Kern: Kernel<
            LinkOpt<Scp>,
            Scp,
            FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>,
            StepSId,
            TpeSInfo,
            Out,
        >,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, StepSId, TpeSInfo>, TypeCodom<Out>>>,
    Splt: Splitter<Xy<RawOpt<Scp::SolShape, StepSId, TpeSInfo>, TypeCodom<Out>>>,
    TypeCodom<Out>: Orderable,
    Out: Outcome,
{
}

// Implementation for Mixed-NoDomain searchspace
impl<Kern, Wght, Splt, Scp, Out>
    SingleSampler<
        BaseSol<SId, LinkOpt<Scp>, TpeSInfo>,
        SId,
        LinkOpt<Scp>,
        Out,
        Scp,
        SimpleObjective<Scp::SolShape, TpeSInfo, Out>,
    > for Tpe<Kern, Wght, Splt, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, Out>
where
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo> + Send + Sync,
    Kern: Kernel<LinkOpt<Scp>, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, Out>
        + Send
        + Sync,
    Wght: Weighter<Xy<Arc<[TypeDom<LinkOpt<Scp>>]>, TypeCodom<Out>>> + Send + Sync,
    Splt: Splitter<Xy<Arc<[TypeDom<LinkOpt<Scp>>]>, TypeCodom<Out>>> + Send + Sync,
    TypeCodom<Out>: Send + Sync + Orderable,
    TypeDom<Scp::Opt>: Send + Sync,
    Kern::KContext: Send + Sync,
    Kern::SContext: Send + Sync,
    Out: Outcome,
{
    fn sample(
        &mut self,
        scp: &Scp,
        _acc: &CompAcc<Scp::SolShape, SId, Self::SInfo, Out>,
    ) -> Scp::SolShape {
        let n_init = self.0.n_init;
        if self.0.point_archive[0].1.size() < n_init {
            let info = TpeSInfo::new(0.0, 0.0, 0.0);
            self.with_rng(|rng| scp.sample_pair(rng, info))
        } else {
            // Split the archive into good and bad, and compute the weights
            let (good, bad) = self.0.splitter.split(&self.0.point_archive[0].1);
            let weights = self.0.weighter.weight(&good, &bad); // Weights for the good and bad points
            let (good_sctx, good_kctx) = Kern::get_context(&good, scp);
            let (bad_sctx, bad_kctx) = Kern::get_context(&bad, scp);

            let (s, acq, gpdf, bpdf) = (0..self.0.n_sample)
                .into_par_iter()
                .map(|_| {
                    let s = self.with_rng(|rng| {
                        self.0
                            .kernel
                            .sample(rng, &good, &good_kctx, &good_sctx, scp)
                    });
                    let kernel = &self.0.kernel;
                    let good_pdf =
                        kernel.compute(&s, &good, &good_kctx, &good_sctx, &weights.good, scp);
                    let bad_pdf = kernel.compute(&s, &bad, &bad_kctx, &bad_sctx, &weights.bad, scp);
                    let acq = acquisition(good_pdf, bad_pdf);
                    (s, acq, good_pdf, bad_pdf)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();

            let info = TpeSInfo::new(acq, gpdf, bpdf);
            scp.new_opt(s, info)
        }
    }

    fn update(
        &mut self,
        x: &CompShape<Scp::SolShape, SId, Self::SInfo, Out>,
        _scp: &Scp,
        _acc: &CompAcc<Scp::SolShape, SId, Self::SInfo, Out>,
    ) {
        let xy = x.get_sopt().xy();
        self.0.point_archive[0].1.add(xy);
    }

    fn sample_apply<F>(
        &mut self,
        f: F,
        scp: &Scp,
        acc: &CompAcc<Scp::SolShape, SId, Self::SInfo, Out>,
    ) -> Scp::SolShape
    where
        F: Fn(Scp::SolShape) -> Scp::SolShape + Send + Sync,
    {
        let sol = self.sample(scp, acc);
        f(sol)
    }
}

impl<Kern, Wght, Splt, Scp, Out, FnState>
    SingleSampler<
        FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>,
        StepSId,
        LinkOpt<Scp>,
        Out,
        Scp,
        SimpleStepped<Scp::SolShape, TpeSInfo, Out, FnState>,
    > for Tpe<Kern, Wght, Splt, Scp, FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, Out>
where
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo> + Send + Sync,
    Kern: Kernel<
            LinkOpt<Scp>,
            Scp,
            FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>,
            StepSId,
            TpeSInfo,
            Out,
        > + Send
        + Sync,
    Wght: Weighter<Xy<Arc<[TypeDom<LinkOpt<Scp>>]>, TypeCodom<Out>>> + Send + Sync,
    Splt: Splitter<Xy<Arc<[TypeDom<LinkOpt<Scp>>]>, TypeCodom<Out>>> + Send + Sync,
    TypeCodom<Out>: Send + Sync + Orderable,
    TypeDom<Scp::Opt>: Send + Sync,
    Kern::KContext: Send + Sync,
    Kern::SContext: Send + Sync,
    Out: FidOutcome,
    FnState: FuncState,
    Scp::SolShape: HasStep + HasFidelity,
    FCompShape<Scp, Out, TpeSInfo>: HasStep + HasFidelity,
{
    /// Samples a new point in the search space using the TPE acquisition function.
    /// If the number of points in the current archive is less than `n_init`,
    /// samples a random point.
    /// Otherwise, samples `n_sample` points from the KDEs
    /// of the good and bad points, computes their acquisition values,
    /// and returns the point with the highest acquisition value.
    fn sample(
        &mut self,
        scp: &Scp,
        _acc: &CompAcc<Scp::SolShape, StepSId, Self::SInfo, Out>,
    ) -> Scp::SolShape {
        let n_init = self.0.n_init;
        if self.0.point_archive[self.0.current_archive].1.size() < n_init {
            let info = TpeSInfo::new(0.0, 0.0, 0.0);
            self.with_rng(|rng| scp.sample_pair(rng, info))
        } else {
            // Split the archive into good and bad, and compute the weights
            let (good, bad) = self.0.splitter.split(&self.0.point_archive[0].1);
            let weights = self.0.weighter.weight(&good, &bad); // Weights for the good and bad points
            let (good_sctx, good_kctx) = Kern::get_context(&good, scp);
            let (bad_sctx, bad_kctx) = Kern::get_context(&bad, scp);

            let (s, acq, gpdf, bpdf) = (0..self.0.n_sample)
                .into_par_iter()
                .map(|_| {
                    let s = self.with_rng(|rng| {
                        self.0
                            .kernel
                            .sample(rng, &good, &good_kctx, &good_sctx, scp)
                    });
                    let kernel = &self.0.kernel;
                    let good_pdf =
                        kernel.compute(&s, &good, &good_kctx, &good_sctx, &weights.good, scp);
                    let bad_pdf = kernel.compute(&s, &bad, &bad_kctx, &bad_sctx, &weights.bad, scp);
                    let acq = acquisition(good_pdf, bad_pdf);
                    (s, acq, good_pdf, bad_pdf)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();

            let info = TpeSInfo::new(acq, gpdf, bpdf);
            scp.new_opt(s, info)
        }
    }

    /// Updates the point archive with the new point `x`.
    /// If the fidelity of `x` is equal to the current fidelity, adds it to the current archive.
    /// If the fidelity of `x` is higher than the current fidelity,
    /// checks if an archive for this fidelity already exists.
    /// If it does, adds `x` to that archive.
    /// If it doesn't, creates a new archive for this fidelity and adds `x` to it.
    /// If the number of points in the new archive exceeds `n_init`,
    /// removes all archives with lower fidelities and updates the current archive and fidelity to the new ones.
    /// This way, it mimics the behavior of BOHB, where only the points with the highest fidelity
    /// are used for sampling if there are enough of them, otherwise points with lower fidelities are used.
    fn update(
        &mut self,
        x: &CompShape<Scp::SolShape, StepSId, Self::SInfo, Out>,
        _scp: &Scp,
        _acc: &CompAcc<Scp::SolShape, StepSId, Self::SInfo, Out>,
    ) {
        let fidelity = x.fidelity().0;

        if fidelity == self.0.current_fidelity {
            let xy = x.get_sopt().xy();
            self.0.point_archive[self.0.current_archive].1.add(xy);
        } else if fidelity > self.0.current_fidelity {
            let where_is = self.0.point_archive.iter().position(|f| f.0 == fidelity);
            // If the fidelity was already present in the archive.
            if let Some(idx) = where_is {
                let xy = x.get_sopt().xy();
                self.0.point_archive[idx].1.add(xy);

                // If the number of points at this fidelity exceeds the threshold, remove all previous fidelity points.
                if self.0.point_archive[idx].1.size() >= self.0.n_init {
                    self.0.point_archive.retain(|i| i.0 >= fidelity);
                    let i = self
                        .0
                        .point_archive
                        .iter()
                        .position(|f| f.0 == fidelity)
                        .unwrap();
                    self.0.current_archive = i;
                    self.0.current_fidelity = fidelity;
                }
            } else {
                let xy = x.get_sopt().xy();
                self.0
                    .point_archive
                    .push((fidelity, OrderedArchive::new(xy)));
            }
        }
        // Else we ignore as an archive with higher accuracy has already reached the threshold
    }

    fn sample_apply<F>(
        &mut self,
        f: F,
        scp: &Scp,
        acc: &CompAcc<Scp::SolShape, StepSId, Self::SInfo, Out>,
    ) -> Scp::SolShape
    where
        F: Fn(Scp::SolShape) -> Scp::SolShape + Send + Sync,
    {
        let sol = <Tpe<_, _, _, _, _, _, _> as SingleSampler<
            _,
            _,
            _,
            Out,
            _,
            SimpleStepped<Scp::SolShape, _, Out, FnState>,
        >>::sample(self, scp, acc);
        f(sol)
    }
}
