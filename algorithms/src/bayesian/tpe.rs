
use tantale_core::{BaseSol, CSVWritable, Codomain, Criteria, FidOutcome, FidelitySol, FuncState, HasFidelity, HasStep, Id, IntoComputedShape, LinkOpt, OptState, Optimizer, Outcome, SId, Searchspace, SequentialOptimizer, SingleCodomain, SolInfo, SolutionShape, Step, StepSId, Uncomputed, domain::codomain::ElemSingleCodomain, solution::{computed::Xy, shape::RawOpt}};
use crate::{bayesian::{kernel::Kernel, splitter::Splitter, weighter::Weighter}, utils::{BCompAcc, BCompShape, FCompAcc, FCompShape, OrdArchive, SimpleObjective, SimpleStepped}};

use std::{cell::RefCell, sync::Arc};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};


thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(rand::make_rng());
}

/// Creates a codomain for Tree-structured Parzen optimization.
///
/// Constructs a [`SingleCodomain`] from a single-objective
/// [`Criteria`].
///
/// # Arguments
///
/// * `extractor` - A [`Criteria`] defining how to extract the
///   optimization objective from the [`Outcome`].
pub fn codomain<Cod, Out>(extractor: Criteria<Out>) -> Cod
where
    Cod: Codomain<Out> + From<SingleCodomain<Out>>,
    Out: Outcome,
{
    let out = SingleCodomain {
        y_criteria: extractor,
    };
    out.into()
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

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Wght: Serialize, Splt: Serialize",
    deserialize = "Wght: for<'a> Deserialize<'a>, Splt: for<'a> Deserialize<'a>",
))]
pub struct TpeState<Kern, Wght, Splt, Scp, S, SolId, Cod, Out> 
where
    S: Uncomputed<SolId, Scp::Opt, TpeSInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, TpeSInfo>,
    Scp: Searchspace<S, SolId, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo, Cod, Out>,
    Wght: Weighter<Xy<S::Raw, Cod::TypeCodom>>,
    Splt: Splitter,
    SolId: Id,
    Cod: Codomain<Out>,
    Cod::TypeCodom: Ord,
    Out: Outcome,
{
    pub n_init: usize,
    pub n_sample: usize,
    pub kernel: Kern,
    pub weighter: Wght,
    pub splitter: Splt,
    pub point_archive: Vec<OrdArchive<Xy<S::Raw, Cod::TypeCodom>>>,
}

impl<Kern, Wght, Splt, Scp, S, SolId, Cod, Out> OptState for TpeState<Kern, Wght, Splt, Scp, S, SolId, Cod, Out> 
where
    S: Uncomputed<SolId, Scp::Opt, TpeSInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, TpeSInfo>,
    Scp: Searchspace<S, SolId, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo, Cod, Out>,
    Wght: Weighter<Xy<S::Raw, Cod::TypeCodom>>,
    Splt: Splitter,
    SolId: Id,
    Cod: Codomain<Out>,
    Cod::TypeCodom: Ord,
    Out: Outcome,
{

}

pub struct Tpe<Kern, Wght, Splt, Scp, S, SolId, Cod, Out>(pub TpeState<Kern, Wght, Splt, Scp, S, SolId, Cod, Out>)
where
    Scp: Searchspace<S, SolId, TpeSInfo>,
    S: Uncomputed<SolId, Scp::Opt, TpeSInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo, Cod, Out>,
    Wght: Weighter<Xy<S::Raw, Cod::TypeCodom>>,
    Splt: Splitter,
    SolId: Id,
    Cod: Codomain<Out>,
    Cod::TypeCodom: Ord,
    Out: Outcome;

impl<Kern, Wght, Splt, Scp, S, SolId, Cod, Out> Tpe<Kern, Wght, Splt, Scp, S, SolId, Cod, Out>
where
    Scp: Searchspace<S, SolId, TpeSInfo>,
    S: Uncomputed<SolId, Scp::Opt, TpeSInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo, Cod, Out>,
    Wght: Weighter<Xy<S::Raw, Cod::TypeCodom>>,
    Splt: Splitter,
    SolId: Id,
    Cod: Codomain<Out>,
    Cod::TypeCodom: Ord,
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
            point_archive: vec![OrdArchive::default()],
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
    for Tpe<Kern, Wght, Splt, Scp, BaseSol<SId, Scp::Opt, TpeSInfo>, SId, SingleCodomain<Out>, Out>
where
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo>,
    Kern: Kernel<LinkOpt<Scp>, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, SingleCodomain<Out>, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, SId, TpeSInfo>, ElemSingleCodomain>>,
    Splt: Splitter,
    Out: Outcome,
{
    type State = TpeState<Kern, Wght, Splt, Scp, BaseSol<SId, Scp::Opt, TpeSInfo>, SId, SingleCodomain<Out>, Out>;
    type Cod = SingleCodomain<Out>;
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
    SequentialOptimizer<BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, LinkOpt<Scp>, Out, Scp, SimpleObjective<Scp::SolShape, TpeSInfo, Out>,>
    for Tpe<Kern, Wght, Splt, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, SingleCodomain<Out>, Out>
where
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo>,
    Kern: Kernel<LinkOpt<Scp>, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, SingleCodomain<Out>, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, SId, TpeSInfo>, ElemSingleCodomain>>,
    Splt: Splitter,
    Out: Outcome,
{
    fn step(
        &mut self,
        x: Option<BCompShape<Scp, Out, TpeSInfo, Self::Cod>>,
        scp: &Scp,
        _acc: &BCompAcc<Scp, Out, TpeSInfo, Self::Cod>,
    ) -> Scp::SolShape 
    {
        if let Some(point) = x {
            self.0.point_archive[0].add(point.get_sopt().xy());
        }

        let n_init = self.0.n_init;
        if self.0.point_archive[0].size() < n_init {
            let info = TpeSInfo::new(0.0, 0.0, 0.0);
            self.with_rng(|rng| scp.sample_pair(rng, info))
        } else {
            let points = &self.0.point_archive[0].points;
            let ctx = self.0.kernel.prepare(points, scp);
            let (good, bad) = self.0.splitter.split(&self.0.point_archive[0]);
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


impl<Kern, Wght, Splt, Scp, Out> 
    Optimizer<FidelitySol<StepSId, Scp::Opt, TpeSInfo>, StepSId, Scp::Opt, Out, Scp> 
    for Tpe<Kern, Wght, Splt, Scp, FidelitySol<StepSId, Scp::Opt, TpeSInfo>, StepSId, SingleCodomain<Out>, Out>
where
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo>,
    Kern: Kernel<LinkOpt<Scp>, Scp, FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo, SingleCodomain<Out>, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, StepSId, TpeSInfo>, ElemSingleCodomain>>,
    Splt: Splitter,
    Out: Outcome,
{
    type State = TpeState<Kern, Wght, Splt, Scp, FidelitySol<StepSId, Scp::Opt, TpeSInfo>, StepSId, SingleCodomain<Out>, Out>;
    type Cod = SingleCodomain<Out>;
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


impl<Kern, Wght, Splt, Scp, Out, FnState> 
    SequentialOptimizer<FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, LinkOpt<Scp>, Out, Scp, SimpleStepped<Scp::SolShape, TpeSInfo, Out, FnState>>
    for Tpe<Kern, Wght, Splt, Scp, FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, SingleCodomain<Out>, Out>
where
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo>,
    Kern: Kernel<LinkOpt<Scp>, Scp, FidelitySol<StepSId, LinkOpt<Scp>, TpeSInfo>, StepSId, TpeSInfo, SingleCodomain<Out>, Out>,
    Wght: Weighter<Xy<RawOpt<Scp::SolShape, StepSId, TpeSInfo>, ElemSingleCodomain>>,
    Splt: Splitter,
    Out: FidOutcome,
    FnState: FuncState,
    Scp::SolShape: HasStep + HasFidelity,
    FCompShape<Scp, Out, TpeSInfo, SingleCodomain<Out>>: HasStep + HasFidelity,
{
    fn step(
        &mut self,
        x: Option<FCompShape<Scp, Out, TpeSInfo, Self::Cod>>,
        scp: &Scp,
        _acc: &FCompAcc<Scp, Out, TpeSInfo, Self::Cod>,
    ) -> Scp::SolShape 
    {   
        if let Some(comp) = x {
            self.0.point_archive[0].add(comp.get_sopt().xy());
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
            if self.0.point_archive[0].size() < n_init {
                let info = TpeSInfo::new(0.0, 0.0, 0.0);
                self.with_rng(|rng| scp.sample_pair(rng, info))
            } else {
                let points = &self.0.point_archive[0].points;
                let ctx = self.0.kernel.prepare(points, scp);
                let (good, bad) = self.0.splitter.split(&self.0.point_archive[0]);
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