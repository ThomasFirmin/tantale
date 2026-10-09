use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tantale_core::{Domain, GridDom, HasVariables, HasX, Int, Mixed, MixedTypeDom, Nat, Problem, Real, Searchspace, Unit, Xy, aliases::{CodElem, XOpt}, domain::grid::GridBounds};

use crate::{bayesian::{AitchisonAitkenKernel, GaussianKernel, MixedKernel, Multivariate, kernel::{Kernel, KernelFunc}}};

/// A type alias for `Predictor`, representing the Y values in Nadara-Watson regression.
pub type Predictor = Vec<f64>;

/// A trait for Nadaraya-Watson regression models.
pub trait NwRegressor<P, Dom>: Kernel<P, Dom>
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    P: Problem,
    P::Scp: Searchspace<P::Func, Opt = Dom>,
    Dom: Domain,
{
    fn predict(
        &self,
        x: &XOpt<P>,
        y: &Predictor,
        weights: &[f64],
        archive: &[&Xy<XOpt<P>, CodElem<P>>],
        context: &[Self::Context],
        scp: &P::Scp,
    ) -> f64;
}

impl<P> NwRegressor<P, Mixed> for Multivariate
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    P: Problem<XOpt = Arc<[MixedTypeDom]>>,
    P::Scp: Searchspace<P::Func, Opt = Mixed> + HasVariables,
{
    fn predict(
        &self,
        x: &XOpt<P>,
        y: &Predictor,
        weights: &[f64],
        archive: &[&Xy<XOpt<P>, CodElem<P>>],
        context: &[Self::Context],
        scp: &<P as Problem>::Scp,
    ) -> f64
    {
        let sum: (f64, f64) = archive
            .iter()
            .zip(y.iter())
            .zip(weights)
            .zip(context.iter())
            .map(|(((comp, pred), w), ctx)| {
                let product: f64 = x.iter()
                    .zip(comp.as_ref().ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), ctx)| {
                        MixedKernel::compute(x1, x2, ctx, dom)
                    })
                    .product();
                (product * pred * w, product * w)
            })
            .fold((0.0, 0.0), |(sum_kw, sum_k), (prod_w, prod)| {
                (sum_kw + prod_w, sum_k + prod)
            });
        sum.0 / sum.1
    }
}

impl<P> NwRegressor<P, Real> for Multivariate
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    P: Problem<XOpt = Arc<[f64]>>,
    P::Scp: Searchspace<P::Func, Opt = Real> + HasVariables,
{
    fn predict(
        &self,
        x: &XOpt<P>,
        y: &Predictor,
        weights: &[f64],
        archive: &[&Xy<XOpt<P>, CodElem<P>>],
        context: &[Self::Context],
        scp: &<P as Problem>::Scp,
    ) -> f64
    {
        let sum: (f64, f64) = archive
            .iter()
            .zip(y.iter())
            .zip(weights)
            .zip(context.iter())
            .map(|(((comp, pred), w), ctx)| {
                let product: f64 = x.iter()
                    .zip(comp.as_ref().ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), ctx)| {
                        GaussianKernel::compute(x1, x2, ctx, dom)
                    })
                    .product();
                (product * pred * w, product * w)
            })
            .fold((0.0, 0.0), |(sum_kw, sum_k), (prod_w, prod)| {
                (sum_kw + prod_w, sum_k + prod)
            });
        sum.0 / sum.1
    }
}

impl<P> NwRegressor<P, Int> for Multivariate
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    P: Problem<XOpt = Arc<[i64]>>,
    P::Scp: Searchspace<P::Func, Opt = Int> + HasVariables,
{
    fn predict(
        &self,
        x: &XOpt<P>,
        y: &Predictor,
        weights: &[f64],
        archive: &[&Xy<XOpt<P>, CodElem<P>>],
        context: &[Self::Context],
        scp: &<P as Problem>::Scp,
    ) -> f64
    {
        let sum: (f64, f64) = archive
            .iter()
            .zip(y.iter())
            .zip(weights)
            .zip(context.iter())
            .map(|(((comp, pred), w), ctx)| {
                let product: f64 = x.iter()
                    .zip(comp.as_ref().ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), ctx)| {
                        GaussianKernel::compute(x1, x2, ctx, dom)
                    })
                    .product();
                (product * pred * w, product * w)
            })
            .fold((0.0, 0.0), |(sum_kw, sum_k), (prod_w, prod)| {
                (sum_kw + prod_w, sum_k + prod)
            });
        sum.0 / sum.1
    }
}

impl<P> NwRegressor<P, Nat> for Multivariate
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    P: Problem<XOpt = Arc<[u64]>>,
    P::Scp: Searchspace<P::Func, Opt = Nat> + HasVariables,
{
    fn predict(
        &self,
        x: &XOpt<P>,
        y: &Predictor,
        weights: &[f64],
        archive: &[&Xy<XOpt<P>, CodElem<P>>],
        context: &[Self::Context],
        scp: &<P as Problem>::Scp,
    ) -> f64
    {
        let sum: (f64, f64) = archive
            .iter()
            .zip(y.iter())
            .zip(weights)
            .zip(context.iter())
            .map(|(((comp, pred), w), ctx)| {
                let product: f64 = x.iter()
                    .zip(comp.as_ref().ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), ctx)| {
                        GaussianKernel::compute(x1, x2, ctx, dom)
                    })
                    .product();
                (product * pred * w, product * w)
            })
            .fold((0.0, 0.0), |(sum_kw, sum_k), (prod_w, prod)| {
                (sum_kw + prod_w, sum_k + prod)
            });
        sum.0 / sum.1
    }
}

impl<P> NwRegressor<P, Unit> for Multivariate
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    P: Problem<XOpt = Arc<[f64]>>,
    P::Scp: Searchspace<P::Func, Opt = Unit> + HasVariables,
{
    fn predict(
        &self,
        x: &XOpt<P>,
        y: &Predictor,
        weights: &[f64],
        archive: &[&Xy<XOpt<P>, CodElem<P>>],
        context: &[Self::Context],
        scp: &<P as Problem>::Scp,
    ) -> f64
    {
        let sum: (f64, f64) = archive
            .iter()
            .zip(y.iter())
            .zip(weights)
            .zip(context.iter())
            .map(|(((comp, pred), w), ctx)| {
                let product: f64 = x.iter()
                    .zip(comp.as_ref().ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), ctx)| {
                        GaussianKernel::compute(x1, x2, ctx, dom)
                    })
                    .product();
                (product * pred * w, product * w)
            })
            .fold((0.0, 0.0), |(sum_kw, sum_k), (prod_w, prod)| {
                (sum_kw + prod_w, sum_k + prod)
            });
        sum.0 / sum.1
    }
}

impl<P, G> NwRegressor<P, GridDom<G>> for Multivariate
where
    Self: Clone + Sized + Serialize + for<'a> Deserialize<'a>,
    P: Problem<XOpt = Arc<[G]>>,
    P::Scp: Searchspace<P::Func, Opt = GridDom<G>> + HasVariables,
    G: GridBounds,
{
    fn predict(
        &self,
        x: &XOpt<P>,
        y: &Predictor,
        weights: &[f64],
        archive: &[&Xy<XOpt<P>, CodElem<P>>],
        context: &[Self::Context],
        scp: &<P as Problem>::Scp,
    ) -> f64
    {
        let sum: (f64, f64) = archive
            .iter()
            .zip(y.iter())
            .zip(weights)
            .zip(context.iter())
            .map(|(((comp, pred), w), ctx)| {
                let product: f64 = x.iter()
                    .zip(comp.as_ref().ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), ctx)| {
                        AitchisonAitkenKernel::compute(x1, x2, ctx, dom)
                    })
                    .product();
                (product * pred * w, product * w)
            })
            .fold((0.0, 0.0), |(sum_kw, sum_k), (prod_w, prod)| {
                (sum_kw + prod_w, sum_k + prod)
            });
        sum.0 / sum.1
    }
}
