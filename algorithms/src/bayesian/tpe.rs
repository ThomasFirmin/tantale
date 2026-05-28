
use tantale_core::{BaseSol, CSVWritable, CompAcc, CompShape, FidOutcome, FidelitySol, FuncState, HasFidelity, HasStep, Id, IntoComputedShape, LinkOpt, OptState, Optimizer, Outcome, SId, Sampler, Searchspace, SingleOptimizer, SingleSampler, SolInfo, SolutionShape, Step, StepSId, Uncomputed, domain::codomain::TypeCodom, solution::{computed::Xy, shape::RawOpt}};
use crate::{bayesian::{kernel::Kernel, splitter::Splitter, weighter::Weighter}, utils::{BCompAcc, BCompShape, FCompAcc, FCompShape, OrdArchive, SimpleObjective, SimpleStepped}};

use std::{cell::RefCell, sync::Arc};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};


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

impl CSVWritable<(),()> for TpeSInfo {
    fn header(_elem: &()) -> Vec<String> {
        vec!["acquisition".to_string(), "good_bdf".to_string(), "bad_pdf".to_string()]
    }
    
    fn write(&self, _comp: &()) -> Vec<String> {
        vec![format!("{},{},{}", self.acquisition, self.good_bdf, self.bad_pdf)]
    }
}

type JointFidelityArchive<Raw, CodElem> = Vec<(f64, OrdArchive<Xy<Raw, CodElem>>)>;

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Wght: Serialize, Splt: Serialize",
    deserialize = "Wght: for<'a> Deserialize<'a>, Splt: for<'a> Deserialize<'a>",
))]
pub struct TpeState<Kern, Wght, Splt, Scp, S, SolId, Out> 
where
    S: Uncomputed<SolId, Scp::Opt, TpeSInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, TpeSInfo>,
    Scp: Searchspace<S, SolId, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo, Out>,
    Wght: Weighter<Xy<S::Raw, TypeCodom<Out>>>,
    Splt: Splitter,
    SolId: Id,
    TypeCodom<Out>: Ord,
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

impl<Kern, Wght, Splt, Scp, S, SolId, Out> OptState for TpeState<Kern, Wght, Splt, Scp, S, SolId, Out> 
where
    S: Uncomputed<SolId, Scp::Opt, TpeSInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, TpeSInfo>,
    Scp: Searchspace<S, SolId, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo, Out>,
    Wght: Weighter<Xy<S::Raw, TypeCodom<Out>>>,
    Splt: Splitter,
    SolId: Id,
    TypeCodom<Out>: Ord,
    Out: Outcome,
{

}

pub struct Tpe<Kern, Wght, Splt, Scp, S, SolId, Out>(pub TpeState<Kern, Wght, Splt, Scp, S, SolId, Out>)
where
    Scp: Searchspace<S, SolId, TpeSInfo>,
    S: Uncomputed<SolId, Scp::Opt, TpeSInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo, Out>,
    Wght: Weighter<Xy<S::Raw, TypeCodom<Out>>>,
    Splt: Splitter,
    SolId: Id,
    TypeCodom<Out>: Ord,
    Out: Outcome;

impl<Kern, Wght, Splt, Scp, S, SolId, Out> Tpe<Kern, Wght, Splt, Scp, S, SolId, Out>
where
    Scp: Searchspace<S, SolId, TpeSInfo>,
    S: Uncomputed<SolId, Scp::Opt, TpeSInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo, Out>,
    Wght: Weighter<Xy<S::Raw, TypeCodom<Out>>>,
    Splt: Splitter,
    SolId: Id,
    TypeCodom<Out>: Ord,
    Out: Outcome,
{
    pub fn new(n_init: usize, n_sample:usize, kernel: Kern, weighter: Wght, splitter: Splt) -> Self {
        assert!(n_init > 0, "n_init must be greater than 0");
        assert!(n_sample > 0, "n_sample must be greater than 0");
        Tpe(TpeState {
            n_init,
            n_sample,
            kernel,
            weighter,
            splitter,
            point_archive: vec![(0.0, OrdArchive::default())],
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
    Splt: Splitter,
    TypeCodom<Out>: Ord,
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
    Kern: Kernel<LinkOpt<Scp>, Scp, FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, StepSId, TpeSInfo>, TypeCodom<Out>>>,
    Splt: Splitter,
    TypeCodom<Out>: Ord,
    Out: Outcome,
{
    type State = TpeState<Kern, Wght, Splt, Scp, FidelitySol<StepSId, Scp::Opt, TpeSInfo>, StepSId, Out>;
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
    SingleOptimizer<BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, LinkOpt<Scp>, Out, Scp, SimpleObjective<Scp::SolShape, TpeSInfo, Out>,>
    for Tpe<Kern, Wght, Splt, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, Out>
where
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo>,
    Kern: Kernel<LinkOpt<Scp>, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, SId, TpeSInfo>, TypeCodom<Out>>>,
    Splt: Splitter,
    TypeCodom<Out>: Ord,
    Out: Outcome,
{
    fn step(
        &mut self,
        x: Option<BCompShape<Scp, Out, TpeSInfo>>,
        scp: &Scp,
        _acc: &BCompAcc<Scp, Out, TpeSInfo>,
    ) -> Scp::SolShape 
    {
        if let Some(point) = x {
            self.0.point_archive[0].1.add(point.get_sopt().xy());
        }

        let n_init = self.0.n_init;
        if self.0.point_archive[0].1.size() < n_init {
            let info = TpeSInfo::new(0.0, 0.0, 0.0);
            self.with_rng(|rng| scp.sample_pair(rng, info))
        } else {
            let points = &self.0.point_archive[0].1.points;
            let ctx = self.0.kernel.prepare(points, scp);
            let (good, bad) = self.0.splitter.split(&self.0.point_archive[0].1);
            let weights = self.0.weighter.weight(good, bad);
            let (s, acq, gpdf, bpdf) = (0..self.0.n_sample).map(
                |_|
                {
                    let s = self.with_rng(
                        |rng| self.0.kernel.sample(rng, points, scp, &ctx)
                    );
                    let kernel = &self.0.kernel;
                    let good_pdf = kernel.compute(
                        &s, 
                        points, 
                        &weights.good,
                        scp,
                        &ctx
                    );
                    let bad_pdf = kernel.compute(
                        &s, 
                        points, 
                        &weights.bad,
                        scp,
                        &ctx
                    );
                    let acq = acquisition(good_pdf, bad_pdf);
                    (s, acq, good_pdf, bad_pdf)
                }
            ).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();

            let info = TpeSInfo::new(acq, gpdf, bpdf);
            scp.new_opt(s, info)
        }
    }
}


impl<Kern, Wght, Splt, Scp, Out, FnState> 
    SingleOptimizer<FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, LinkOpt<Scp>, Out, Scp, SimpleStepped<Scp::SolShape, TpeSInfo, Out, FnState>>
    for Tpe<Kern, Wght, Splt, Scp, FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, Out>
where
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo>,
    Kern: Kernel<LinkOpt<Scp>, Scp, FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, StepSId, TpeSInfo>, TypeCodom<Out>>>,
    Splt: Splitter,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    FnState: FuncState,
    Scp::SolShape: HasStep + HasFidelity,
    FCompShape<Scp, Out, TpeSInfo>: HasStep + HasFidelity,
{
    fn step(
        &mut self,
        x: Option<FCompShape<Scp, Out, TpeSInfo>>,
        scp: &Scp,
        _acc: &FCompAcc<Scp, Out, TpeSInfo>,
    ) -> Scp::SolShape 
    {   
        if let Some(comp) = x {
            self.0.point_archive[0].1.add(comp.get_sopt().xy());
            let (pair, _): (Scp::SolShape, _) = IntoComputedShape::extract(comp);
            match pair.step() {
                Step::Pending => {
                    unreachable!(
                        "A pending SolShape, should not be passed to RandomSearch step."
                    )
                }
                Step::Partially(_) => pair,
                _ => {
                    let info = TpeSInfo::new(0.0, 0.0, 0.0);
                    self.with_rng(|rng| scp.sample_pair(rng, info))
                },
            }
        } else {
            let n_init = self.0.n_init;
            if self.0.point_archive[0].1.size() < n_init {
                let info = TpeSInfo::new(0.0, 0.0, 0.0);
                self.with_rng(|rng| scp.sample_pair(rng, info))
            } else {
                let points = &self.0.point_archive[0].1.points;
                let ctx = self.0.kernel.prepare(points, scp);
                let (good, bad) = self.0.splitter.split(&self.0.point_archive[0].1);
                let weights = self.0.weighter.weight(good, bad);
                let (s, acq, gpdf, bpdf) = (0..self.0.n_sample).map(
                    |_|
                    {
                        let s = self.with_rng(
                            |rng| self.0.kernel.sample(rng, points, scp, &ctx)
                        );
                        let kernel = &self.0.kernel;
                        let good_pdf = kernel.compute(
                            &s, 
                            points, 
                            &weights.good,
                            scp,
                            &ctx
                        );
                        let bad_pdf = kernel.compute(
                            &s, 
                            points, 
                            &weights.bad,
                            scp,
                            &ctx
                        );
                        let acq = acquisition(good_pdf, bad_pdf);
                        (s, acq, good_pdf, bad_pdf)
                    }
                ).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();

                let info = TpeSInfo::new(acq, gpdf, bpdf);
                scp.new_opt(s, info)
            }
        }
    }
}







impl<Kern, Wght, Splt, Scp, Out> 
    Sampler<BaseSol<SId, Scp::Opt, TpeSInfo>, SId, Scp::Opt, Out, Scp> 
    for Tpe<Kern, Wght, Splt, Scp, BaseSol<SId, Scp::Opt, TpeSInfo>, SId, Out>
where
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo>,
    Kern: Kernel<LinkOpt<Scp>, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, SId, TpeSInfo>, TypeCodom<Out>>>,
    Splt: Splitter,
    TypeCodom<Out>: Ord,
    Out: Outcome,
{
    
}

impl<Kern, Wght, Splt, Scp, Out> 
    Sampler<FidelitySol<StepSId, Scp::Opt, TpeSInfo>, StepSId, Scp::Opt, Out, Scp> 
    for Tpe<Kern, Wght, Splt, Scp, FidelitySol<StepSId, Scp::Opt, TpeSInfo>, StepSId, Out>
where
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo>,
    Kern: Kernel<LinkOpt<Scp>, Scp, FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, StepSId, TpeSInfo>, TypeCodom<Out>>>,
    Splt: Splitter,
    TypeCodom<Out>: Ord,
    Out: Outcome,
{

}




// Implementation for Mixed-NoDomain searchspace
impl<Kern, Wght, Splt, Scp, Out> 
    SingleSampler<BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, LinkOpt<Scp>, Out, Scp, SimpleObjective<Scp::SolShape, TpeSInfo, Out>,>
    for Tpe<Kern, Wght, Splt, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, Out>
where
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo>,
    Kern: Kernel<LinkOpt<Scp>, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, SId, TpeSInfo>, TypeCodom<Out>>>,
    Splt: Splitter,
    TypeCodom<Out>: Ord,
    Out: Outcome,
{
    fn sample(&mut self, scp: &Scp, _acc: &CompAcc<Scp::SolShape, SId, Self::SInfo, Out>) -> Scp::SolShape {
        let n_init = self.0.n_init;
        if self.0.point_archive[0].1.size() < n_init {
            let info = TpeSInfo::new(0.0, 0.0, 0.0);
            self.with_rng(|rng| scp.sample_pair(rng, info))
        } else {
            let points = &self.0.point_archive[0].1.points;
            let ctx = self.0.kernel.prepare(points, scp);
            let (good, bad) = self.0.splitter.split(&self.0.point_archive[0].1);
            let weights = self.0.weighter.weight(good, bad);
            let (s, acq, gpdf, bpdf) = (0..self.0.n_sample).map(
                |_|
                {
                    let s = self.with_rng(
                        |rng| self.0.kernel.sample(rng, points, scp, &ctx)
                    );
                    let kernel = &self.0.kernel;
                    let good_pdf = kernel.compute(
                        &s, 
                        points, 
                        &weights.good,
                        scp,
                        &ctx
                    );
                    let bad_pdf = kernel.compute(
                        &s, 
                        points, 
                        &weights.bad,
                        scp,
                        &ctx
                    );
                    let acq = acquisition(good_pdf, bad_pdf);
                    (s, acq, good_pdf, bad_pdf)
                }
            ).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();

            let info = TpeSInfo::new(acq, gpdf, bpdf);
            scp.new_opt(s, info)
        }
    }
    
    fn update(&mut self, x: &CompShape<Scp::SolShape, SId, Self::SInfo, Out>, _scp: &Scp, _acc: &CompAcc<Scp::SolShape, SId, Self::SInfo, Out>) {
        self.0.point_archive[0].1.add(x.get_sopt().xy());
    }
}


impl<Kern, Wght, Splt, Scp, Out, FnState> 
    SingleSampler<FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, LinkOpt<Scp>, Out, Scp, SimpleStepped<Scp::SolShape, TpeSInfo, Out, FnState>>
    for Tpe<Kern, Wght, Splt, Scp, FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, Out>
where
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo>,
    Kern: Kernel<LinkOpt<Scp>, Scp, FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, StepSId, TpeSInfo>, TypeCodom<Out>>>,
    Splt: Splitter,
    TypeCodom<Out>: Ord,
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
    fn sample(&mut self, scp: &Scp, _acc: &CompAcc<Scp::SolShape, StepSId, Self::SInfo, Out>) -> Scp::SolShape {
        let n_init = self.0.n_init;
        if self.0.point_archive[self.0.current_archive].1.size() < n_init {
            let info = TpeSInfo::new(0.0, 0.0, 0.0);
            self.with_rng(|rng| scp.sample_pair(rng, info))
        } else {
            let points = &self.0.point_archive[self.0.current_archive].1.points;
            let ctx = self.0.kernel.prepare(points, scp);
            let (good, bad) = self.0.splitter.split(&self.0.point_archive[self.0.current_archive].1);
            let weights = self.0.weighter.weight(good, bad);
            let (s, acq, gpdf, bpdf) = (0..self.0.n_sample).map(
                |_|
                {
                    let s = self.with_rng(
                        |rng| self.0.kernel.sample(rng, points, scp, &ctx)
                    );
                    let kernel = &self.0.kernel;
                    let good_pdf = kernel.compute(
                        &s, 
                        points, 
                        &weights.good,
                        scp,
                        &ctx
                    );
                    let bad_pdf = kernel.compute(
                        &s, 
                        points, 
                        &weights.bad,
                        scp,
                        &ctx
                    );
                    let acq = acquisition(good_pdf, bad_pdf);
                    (s, acq, good_pdf, bad_pdf)
                }
            ).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();

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
    fn update(&mut self, x: &CompShape<Scp::SolShape, StepSId, Self::SInfo, Out>, _scp: &Scp, _acc: &CompAcc<Scp::SolShape, StepSId, Self::SInfo, Out>) {
        let fidelity = x.fidelity().0;

        if fidelity == self.0.current_fidelity {
            self.0.point_archive[self.0.current_archive].1.add(x.get_sopt().xy());    
        } else if fidelity >= self.0.current_fidelity {
            let where_is = self.0.point_archive.iter().position(|f| f.0 == fidelity);
            // If the fidelity was already present in the archive.
            if let Some(idx) = where_is {
                self.0.point_archive[idx].1.add(x.get_sopt().xy());
                // If the number of points at this fidelity exceeds the threshold, remove all previous fidelity points.
                if self.0.point_archive[idx].1.size() > self.0.n_init {
                    self.0.point_archive.retain(|i| i.0 >= fidelity);
                    let i = self.0.point_archive.iter().position(|f| f.0 == fidelity).unwrap();
                    self.0.current_archive = i;
                    self.0.current_fidelity = fidelity;
                }
            } else{
                self.0.point_archive.push((fidelity, OrdArchive::new(x.get_sopt().xy())));
            }
        }
        // Else we ignore as an archive with higher accuracy has already reached the threshold
    }
}