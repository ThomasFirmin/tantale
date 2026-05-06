
use tantale_core::{BaseSol, CSVWritable, Codomain, CompAcc, CompShape, EmptyInfo, FidelitySol, Id, IntoComputed, LinkOpt, Mixed, Objective, OptInfo, OptState, Optimizer, OptionCompShape, Outcome, RawObj, SId, Searchspace, SequentialOptimizer, Single, SingleCodomain, SolInfo, Solution, SolutionShape, StepSId, has_trait::HasVariables};
use crate::{bayesian::{bandwidth::{self, cat_bw, optuna_bw}, kernel::Kernel, splitter::Splitter, weighter::Weighter}, utils::{BCompAcc, BCompShape, PointArchive, SimpleObjective}};

use std::{cell::RefCell, sync::Arc};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};


thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(rand::make_rng());
}

/// Computes the acquisition function value based on the best and worse KDE-PDF values.
pub fn acquisition(best_pdf: f64, worse_pdf: f64) -> f64 {
    best_pdf / worse_pdf
}

/// Computes the Kernel Density Estimation (KDE) probability density function (PDF) for a given point $x$.
/// 
/// # Arguments
/// * `weights` - A slice of weights associated to each known points. The first weight is typically the weight of the prior.
/// * `prior` - The prior associated to point $x$.
/// * `kernel_values` - A slice of kernel values computed between point $x$ and other known point.
/// 
/// # Returns
/// The KDE-PDF value for point $x$.
/// 
/// $$KDE(x) = w_0 \cdot prior + \sum_{i=1}^{N} w_i \cdot kernel(x, x_i)$$
pub fn kde_pdf(kernel_values: &[f64], weights: &[f64], prior_weight: f64, prior: f64,) -> f64{
    let sum: f64 = kernel_values.iter().zip(weights.iter().skip(1)).map(|(k, w)| k*w).sum();
    prior_weight * prior + sum
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct TpeSInfo {
    pub acquisition: f64,
    pub best_pdf: f64,
    pub worse_pdf: f64,
}

impl TpeSInfo {
    pub fn new(acquisition: f64, best_pdf: f64, worse_pdf: f64) -> Arc<Self> {
        Arc::new(TpeSInfo {
            acquisition,
            best_pdf,
            worse_pdf,
        })
    }
}

impl SolInfo for TpeSInfo {}

impl CSVWritable<(),()> for TpeSInfo {
    fn header(_elem: &()) -> Vec<String> {
        vec!["acquisition".to_string(), "best_pdf".to_string(), "worse_pdf".to_string()]
    }
    
    fn write(&self, _comp: &()) -> Vec<String> {
        vec![format!("{},{},{}", self.acquisition, self.best_pdf, self.worse_pdf)]
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Wght: Serialize, Splt: Serialize",
    deserialize = "Wght: for<'a> Deserialize<'a>, Splt: for<'a> Deserialize<'a>",
))]
pub struct TpeState<Kern, Wght, Splt, Scp, S, SolId, Cod, Out> 
where
    Scp: Searchspace<S, SolId, TpeSInfo>,
    S: Solution<SolId, LinkOpt<Scp>, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo>,
    Wght: Weighter<CompShape<Scp, S, SolId, TpeSInfo, Cod, Out>>,
    Splt: Splitter,
    SolId: Id,
    Cod: Codomain<Out>,
    Out: Outcome,
    CompShape<Scp, S, SolId, TpeSInfo, Cod, Out>: Ord,
{
    pub n_init: usize,
    pub kernel: Kern,
    pub weighter: Wght,
    pub splitter: Splt,
    pub point_archive: Vec<PointArchive<CompShape<Scp, S, SolId, TpeSInfo, Cod, Out>>>,
}

impl<Kern, Wght, Splt, Scp, S, SolId, Cod, Out> OptState for TpeState<Kern, Wght, Splt, Scp, S, SolId, Cod, Out> 
where
    Scp: Searchspace<S, SolId, TpeSInfo>,
    S: Solution<SolId, LinkOpt<Scp>, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo>,
    Wght: Weighter<CompShape<Scp, S, SolId, TpeSInfo, Cod, Out>>,
    Splt: Splitter,
    SolId: Id,
    Cod: Codomain<Out>,
    Out: Outcome,
    CompShape<Scp, S, SolId, TpeSInfo, Cod, Out>: Ord,
{

}

pub struct Tpe<Kern, Wght, Splt, Scp, S, SolId, Cod, Out>(TpeState<Kern, Wght, Splt, Scp, S, SolId, Cod, Out>)
where
    Scp: Searchspace<S, SolId, TpeSInfo>,
    S: Solution<SolId, LinkOpt<Scp>, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo>,
    Wght: Weighter<CompShape<Scp, S, SolId, TpeSInfo, Cod, Out>>,
    Splt: Splitter,
    SolId: Id,
    Cod: Codomain<Out>,
    Out: Outcome,
    CompShape<Scp, S, SolId, TpeSInfo, Cod, Out>: Ord;

impl<Kern, Wght, Splt, Scp, S, SolId, Cod, Out> Tpe<Kern, Wght, Splt, Scp, S, SolId, Cod, Out>
where
    Scp: Searchspace<S, SolId, TpeSInfo>,
    S: Solution<SolId, LinkOpt<Scp>, TpeSInfo>,
    Kern: Kernel<Scp::Opt, Scp, S, SolId, TpeSInfo>,
    Wght: Weighter<CompShape<Scp, S, SolId, TpeSInfo, Cod, Out>>,
    Splt: Splitter,
    SolId: Id,
    Cod: Codomain<Out>,
    Out: Outcome,
    CompShape<Scp, S, SolId, TpeSInfo, Cod, Out>: Ord,
{
    pub fn new(n_init: usize, kernel: Kern, weighter: Wght, splitter: Splt) -> Self {
        Tpe(TpeState {
            n_init,
            kernel,
            weighter,
            splitter,
            point_archive: Vec::new(),
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
    Kern: Kernel<LinkOpt<Scp>, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo>,
    Wght: Weighter<CompShape<Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, SingleCodomain<Out>, Out>>,
    Splt: Splitter,
    Out: Outcome,
    CompShape<Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, SingleCodomain<Out>, Out>: Ord,
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
    Kern: Kernel<LinkOpt<Scp>, Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo>,
    Wght: Weighter<CompShape<Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, SingleCodomain<Out>, Out>>,
    Splt: Splitter,
    Out: Outcome,
    CompShape<Scp, BaseSol<SId, LinkOpt<Scp>, TpeSInfo>, SId, TpeSInfo, SingleCodomain<Out>, Out>: SolutionShape<SId, TpeSInfo> + Ord,
{
    fn step(
        &mut self,
        x: Option<BCompShape<Scp, Out, TpeSInfo, Self::Cod>>,
        scp: &Scp,
        acc: &BCompAcc<Scp, Out, TpeSInfo, Self::Cod>,
    ) -> Scp::SolShape 
    {
        let archive = &mut self.0.point_archive[0];

        if let Some(point) = x {
            archive.add(point);
        }


        if archive.size() < self.0.n_init {
            let info = TpeSInfo::new(0.0, 0.0, 0.0);
            self.with_rng(|rng| scp.sample_pair(rng, info))
        } else {
            let (good, bad) = self.0.splitter.split(archive);
            let weights = self.0.weighter.weight(good, bad);
            let kernel_values = self.0.kernel.compute(s1, archive, searchspace)

            todo!()
        }
    }
}