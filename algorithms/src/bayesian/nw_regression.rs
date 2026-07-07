use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tantale_core::{Domain, GridDom, HasVariables, HasX, Id, Int, Mixed, Nat, Outcome, Real, Searchspace, SolInfo, TypeCodom, Uncomputed, Unit, Xy, domain::{TypeDom, grid::GridBounds}};

use crate::{Univariate, bayesian::{AitchisonAitkenKernel, GaussianKernel, MixedKernel, Multivariate, kernel::{Kernel, KernelFunc}}};

/// A type alias for `Predictor`, representing the Y values in Nadara-Watson regression.
pub type Predictor = Vec<f64>;

/// A trait for Nadaraya-Watson regression models.
pub trait NwRegressor<Dom, Scp, S, SolId, SInfo, Out>: Kernel<Dom, Scp, S, SolId, SInfo, Out>
where
    Self: Sized + Serialize + for<'a> Deserialize<'a>,
    Dom: Domain,
    Scp: Searchspace<S, SolId, SInfo, Opt = Dom>,
    S: Uncomputed<SolId, Dom, SInfo>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome, 
{
    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<S::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64;
}


impl<Scp, S, SolId, SInfo, Out> NwRegressor<Mixed, Scp, S, SolId, SInfo, Out> for Univariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Mixed> + HasVariables,
    S: Uncomputed<SolId, Mixed, SInfo, Raw = Arc<[TypeDom<Mixed>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64 {
        // Compute the kernel and the kernel-yed sum for each dimension, then take the product across dimensions.
        let product= x
            .iter()
            .zip(scp.iter_opt())
            .enumerate()
            .map(|(d, (x1, dom))| {
                archive
                    .iter()
                    .zip(y.iter())
                    .zip(context.iter())
                    .fold((0.0, 0.0), |(sum_kw, sum_k), ((comp, pred), ctx)| {
                        let k = MixedKernel::compute(x1, &comp.ref_x()[d], &ctx[d], dom);
                        (sum_kw + k * pred, sum_k + k)
                    })
            })
            .fold((1.0, 1.0), |(prod_kw, prod_k), (sum_kw, sum_k)| {
                (prod_kw * sum_kw, prod_k * sum_k)
            });

        product.0 / product.1
    }
}

impl<Scp, S, SolId, SInfo, Out> NwRegressor<Real, Scp, S, SolId, SInfo, Out> for Univariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Real> + HasVariables,
    S: Uncomputed<SolId, Real, SInfo, Raw = Arc<[TypeDom<Real>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64 {
        let product: (f64, f64) = x
            .iter()
            .zip(scp.iter_opt())
            .enumerate()
            .map(|(d, (x1, dom))| {
                archive
                    .iter()
                    .zip(y.iter())
                    .zip(context.iter())
                    .fold((0.0, 0.0), |(sum_kw, sum_k), ((comp, pred), ctx)| {
                        let k = GaussianKernel::compute(x1, &comp.ref_x()[d], &ctx[d], dom);
                        (sum_kw + k * pred, sum_k + k)
                    })
            })
            .fold((1.0, 1.0), |(prod_kw, prod_k), (sum_kw, sum_k)| {
                (prod_kw * sum_kw, prod_k * sum_k)
            });
        
        product.0 / product.1
    }
}

impl<Scp, S, SolId, SInfo, Out> NwRegressor<Int, Scp, S, SolId, SInfo, Out> for Univariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Int> + HasVariables,
    S: Uncomputed<SolId, Int, SInfo, Raw = Arc<[TypeDom<Int>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64 {
        let product: (f64,f64) = x
            .iter()
            .zip(scp.iter_opt())
            .enumerate()
            .map(|(d, (x1, dom))| {
                archive
                    .iter()
                    .zip(y.iter())
                    .zip(context.iter())
                    .fold((0.0, 0.0), |(sum_kw, sum_k), ((comp, pred), ctx)| {
                        let k = GaussianKernel::compute(x1, &comp.ref_x()[d], &ctx[d], dom);
                        (sum_kw + k * pred, sum_k + k)
                    })
            })
            .fold((1.0, 1.0), |(prod_kw, prod_k), (sum_kw, sum_k)| {
                (prod_kw * sum_kw, prod_k * sum_k)
            });
        
        product.0 / product.1
    }
}

impl<Scp, S, SolId, SInfo, Out> NwRegressor<Nat, Scp, S, SolId, SInfo, Out> for Univariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Nat> + HasVariables,
    S: Uncomputed<SolId, Nat, SInfo, Raw = Arc<[TypeDom<Nat>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64 {
        let product: (f64,f64) = x
            .iter()
            .zip(scp.iter_opt())
            .enumerate()
            .map(|(d, (x1, dom))| {
                archive
                    .iter()
                    .zip(y.iter())
                    .zip(context.iter())
                    .fold((0.0, 0.0), |(sum_kw, sum_k), ((comp, pred), ctx)| {
                        let k = GaussianKernel::compute(x1, &comp.ref_x()[d], &ctx[d], dom);
                        (sum_kw + k * pred, sum_k + k)
                    })
            })
            .fold((1.0, 1.0), |(prod_kw, prod_k), (sum_kw, sum_k)| {
                (prod_kw * sum_kw, prod_k * sum_k)
            });
        
        product.0 / product.1
    }
}

impl<Scp, S, SolId, SInfo, Out> NwRegressor<Unit, Scp, S, SolId, SInfo, Out> for Univariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Unit> + HasVariables,
    S: Uncomputed<SolId, Unit, SInfo, Raw = Arc<[TypeDom<Unit>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64 {
        let product: (f64,f64) = x
            .iter()
            .zip(scp.iter_opt())
            .enumerate()
            .map(|(d, (x1, dom))| {
                archive
                    .iter()
                    .zip(y.iter())
                    .zip(context.iter())
                    .fold((0.0, 0.0), |(sum_kw, sum_k), ((comp, pred), ctx)| {
                        let k = GaussianKernel::compute(x1, &comp.ref_x()[d], &ctx[d], dom);
                        (sum_kw + k * pred, sum_k + k)
                    })
            })
            .fold((1.0, 1.0), |(prod_kw, prod_k), (sum_kw, sum_k)| {
                (prod_kw * sum_kw, prod_k * sum_k)
            });
        
        product.0 / product.1
    }
}

impl<T, Scp, S, SolId, SInfo, Out> NwRegressor<GridDom<T>, Scp, S, SolId, SInfo, Out> for Univariate
where
    T: GridBounds,
    Scp: Searchspace<S, SolId, SInfo, Opt = GridDom<T>> + HasVariables,
    S: Uncomputed<SolId, GridDom<T>, SInfo, Raw = Arc<[TypeDom<GridDom<T>>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64 {
        let product: (f64,f64) = x
            .iter()
            .zip(scp.iter_opt())
            .enumerate()
            .map(|(d, (x1, dom))| {
                archive
                    .iter()
                    .zip(y.iter())
                    .zip(context.iter())
                    .fold((0.0, 0.0), |(sum_kw, sum_k), ((comp, pred), ctx)| {
                        let k = AitchisonAitkenKernel::compute(x1, &comp.ref_x()[d], &ctx[d], dom);
                        (sum_kw + k * pred, sum_k + k)
                    })
            })
            .fold((1.0, 1.0), |(prod_kw, prod_k), (sum_kw, sum_k)| {
                (prod_kw * sum_kw, prod_k * sum_k)
            });
        
        product.0 / product.1
    }
}

impl<Scp, S, SolId, SInfo, Out> NwRegressor<Mixed, Scp, S, SolId, SInfo, Out> for Multivariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Mixed> + HasVariables,
    S: Uncomputed<SolId, Mixed, SInfo, Raw = Arc<[TypeDom<Mixed>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64 {
        let sum: (f64, f64) = archive
            .iter()
            .zip(y.iter())
            .zip(context.iter())
            .map(|((comp, pred), ctx)| {
                let product: f64 = x.iter()
                    .zip(comp.ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), ctx)| {
                        MixedKernel::compute(x1, x2, ctx, dom)
                    })
                    .product();
                (product * pred, product)
            })
            .fold((0.0, 0.0), |(sum_kw, sum_k), (prod_w, prod)| {
                (sum_kw + prod_w, sum_k + prod)
            });
        sum.0 / sum.1
    }
}

impl<Scp, S, SolId, SInfo, Out> NwRegressor<Real, Scp, S, SolId, SInfo, Out> for Multivariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Real> + HasVariables,
    S: Uncomputed<SolId, Real, SInfo, Raw = Arc<[TypeDom<Real>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64 {
        let sum: (f64, f64) = archive
            .iter()
            .zip(y.iter())
            .zip(context.iter())
            .map(|((comp, pred), ctx)| {
                let product: f64 = x.iter()
                    .zip(comp.ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), ctx)| {
                        GaussianKernel::compute(x1, x2, ctx, dom)
                    })
                    .product();
                (product * pred, product)
            })
            .fold((0.0, 0.0), |(sum_kw, sum_k), (prod_w, prod)| {
                (sum_kw + prod_w, sum_k + prod)
            });
        sum.0 / sum.1
    }
}

impl<Scp, S, SolId, SInfo, Out> NwRegressor<Int, Scp, S, SolId, SInfo, Out> for Multivariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Int> + HasVariables,
    S: Uncomputed<SolId, Int, SInfo, Raw = Arc<[TypeDom<Int>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64 {
        let sum: (f64, f64) = archive
            .iter()
            .zip(y.iter())
            .zip(context.iter())
            .map(|((comp, pred), ctx)| {
                let product: f64 = x.iter()
                    .zip(comp.ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), ctx)| {
                        GaussianKernel::compute(x1, x2, ctx, dom)
                    })
                    .product();
                (product * pred, product)
            })
            .fold((0.0, 0.0), |(sum_kw, sum_k), (prod_w, prod)| {
                (sum_kw + prod_w, sum_k + prod)
            });
        sum.0 / sum.1
    }
}

impl<Scp, S, SolId, SInfo, Out> NwRegressor<Nat, Scp, S, SolId, SInfo, Out> for Multivariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Nat> + HasVariables,
    S: Uncomputed<SolId, Nat, SInfo, Raw = Arc<[TypeDom<Nat>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64 {
        let sum: (f64, f64) = archive
            .iter()
            .zip(y.iter())
            .zip(context.iter())
            .map(|((comp, pred), ctx)| {
                let product: f64 = x.iter()
                    .zip(comp.ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), ctx)| {
                        GaussianKernel::compute(x1, x2, ctx, dom)
                    })
                    .product();
                (product * pred, product)
            })
            .fold((0.0, 0.0), |(sum_kw, sum_k), (prod_w, prod)| {
                (sum_kw + prod_w, sum_k + prod)
            });
        sum.0 / sum.1
    }
}

impl<Scp, S, SolId, SInfo, Out> NwRegressor<Unit, Scp, S, SolId, SInfo, Out> for Multivariate
where
    Scp: Searchspace<S, SolId, SInfo, Opt = Unit> + HasVariables,
    S: Uncomputed<SolId, Unit, SInfo, Raw = Arc<[TypeDom<Unit>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{
    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64 {
        let sum: (f64, f64) = archive
            .iter()
            .zip(y.iter())
            .zip(context.iter())
            .map(|((comp, pred), ctx)| {
                let product: f64 = x.iter()
                    .zip(comp.ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), ctx)| {
                        GaussianKernel::compute(x1, x2, ctx, dom)
                    })
                    .product();
                (product * pred, product)
            })
            .fold((0.0, 0.0), |(sum_kw, sum_k), (prod_w, prod)| {
                (sum_kw + prod_w, sum_k + prod)
            });
        sum.0 / sum.1
    }
}

impl<T, Scp, S, SolId, SInfo, Out> NwRegressor<GridDom<T>, Scp, S, SolId, SInfo, Out> for Multivariate
where
    T: GridBounds,
    Scp: Searchspace<S, SolId, SInfo, Opt = GridDom<T>> + HasVariables,
    S: Uncomputed<SolId, GridDom<T>, SInfo, Raw = Arc<[TypeDom<GridDom<T>>]>>,
    S::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Out: Outcome,
{

    fn predict(
        &self,
        x: &S::Raw,
        y: &Predictor,
        archive: &[&Xy<<S>::Raw, TypeCodom<Out>>],
        context: &[Self::Context],
        scp: &Scp,
    ) -> f64 {
        let sum: (f64, f64) = archive
            .iter()
            .zip(y.iter())
            .zip(context.iter())
            .map(|((comp, pred), ctx)| {
                let product: f64 = x.iter()
                    .zip(comp.ref_x().iter())
                    .zip(scp.iter_opt())
                    .zip(ctx.iter())
                    .map(|(((x1, x2), dom), ctx)| {
                        AitchisonAitkenKernel::compute(x1, x2, ctx, dom)
                    })
                    .product();
                (product * pred, product)
            })
            .fold((0.0, 0.0), |(sum_kw, sum_k), (prod_w, prod)| {
                (sum_kw + prod_w, sum_k + prod)
            });
        sum.0 / sum.1
    }
}
