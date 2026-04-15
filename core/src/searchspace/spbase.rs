//! A basic [`Searchspace`] implementation backed by a boxed slice of [`Var`]s.
//!
//! The [`Sp`] type provides a basic, single-threaded [`Searchspace`] with explicit variable
//! ownership. It mirrors the parallel version in [`SpPar`] and can be created
//! either directly or via conversion from [`SpPar`].
//!
//! For end-to-end usage (including the [`hpo!`](../../../tantale/macros/macro.hpo.html) and [`objective!`](../../../tantale/macros/macro.objective.html)
//! macros), see the module-level examples in [`crate::searchspace`].
//!
//! # Example
//! ```
//! use tantale::core::{Sp, Var, Real, Unit, Uniform};
//!
//! let v1 = Var::<Real, Unit>::new("x", Real::new(0.0, 1.0, Uniform), Unit::new(Uniform));
//! let v2 = Var::<Real, Unit>::new("y", Real::new(-1.0, 1.0, Uniform), Unit::new(Uniform));
//! let sp = Sp { var: vec![v1, v2].into_boxed_slice() };
//!
//! assert_eq!(sp.var.len(), 2);
//! ```
//!
//! # Example: convert to [`SpPar`]
//! ```
//! use tantale::core::{Sp, SpPar, Var, Real, Unit, Uniform};
//!
//! let v1 = Var::<Real, Unit>::new("x", Real::new(0.0, 1.0, Uniform), Unit::new(Uniform));
//! let v2 = Var::<Real, Unit>::new("y", Real::new(-1.0, 1.0, Uniform), Unit::new(Uniform));
//! let sp = Sp { var: vec![v1, v2].into_boxed_slice() };
//! let sp_par: SpPar<_, _> = sp.into();
//!
//! assert_eq!(sp_par.var.len(), 2);
//! ```
//!
//! # Example: sampling pairs with [`Searchspace`]
//! ```
//! use tantale::core::{BaseSol, EmptyInfo, HasId, SId, Searchspace, Sp, Var, Real, Unit, Uniform};
//! use std::sync::Arc;
//!
//! let v1 = Var::<Real, Unit>::new("x", Real::new(0.0, 1.0, Uniform), Unit::new(Uniform));
//! let v2 = Var::<Real, Unit>::new("y", Real::new(-1.0, 1.0, Uniform), Unit::new(Uniform));
//! let sp = Sp { var: vec![v1, v2].into_boxed_slice() };
//! let info = Arc::new(EmptyInfo {});
//! let mut rng = rand::rng();
//!
//! type OptSol = BaseSol<SId, Unit, EmptyInfo>;
//! let pair = <Sp<Real, Unit> as Searchspace<OptSol, SId, EmptyInfo>>::sample_pair(
//!     &sp,
//!     &mut rng,
//!     info.clone(),
//! );
//! let pairs = <Sp<Real, Unit> as Searchspace<OptSol, SId, EmptyInfo>>::vec_sample_pair(
//!     &sp,
//!     &mut rng,
//!     3,
//!     info.clone(),
//! );
//! let obj = <Sp<Real, Unit> as Searchspace<OptSol, SId, EmptyInfo>>::sample_obj(
//!     &sp,
//!     &mut rng,
//!     info,
//! );
//! let _pair_from_obj = <Sp<Real, Unit> as Searchspace<OptSol, SId, EmptyInfo>>::onto_opt(
//!     &sp,
//!     obj,
//! );
//!
//! assert_eq!(pairs.len(), 3);
//! let _ = pair.id();
//! ```

use crate::{
    domain::{
        Domain, NoDomain, PreDomain,
        onto::{LinkTyObj, LinkTyOpt, Linked, OntoDom},
    },
    recorder::csv::{CSVLeftRight, CSVWritable},
    searchspace::{Searchspace, SolInfo, SpPar},
    solution::{
        Id, IntoComputed, Lone, Pair, Solution, Uncomputed,
        shape::{RawObj, RawOpt},
    },
    variable::Var,
};

use rand::prelude::Rng;
use std::sync::Arc;

/// A basic [`Searchspace`] made of a boxed slice of [`Var`].
pub struct Sp<Obj: Domain, Opt: PreDomain> {
    /// Variables that define the [`Objective`](crate::Objective) and [`Optimizer`](crate::Optimizer) domains.
    pub var: Box<[Var<Obj, Opt>]>,
}

/// Link objective and optimizer domains when both sides are present.
impl<Obj: OntoDom<Opt>, Opt: OntoDom<Obj>> Linked for Sp<Obj, Opt> {
    type Obj = Obj;
    type Opt = Opt;
}

/// Link objective and optimizer domains (objective domains with itself) when using [`NoDomain`].
impl<Obj: Domain> Linked for Sp<Obj, NoDomain> {
    type Obj = Obj;
    type Opt = Obj;
}

/// Sequential [`Searchspace`] implementation for paired objective/optimizer domains.
impl<SolOpt, SolId, Obj, Opt, SInfo> Searchspace<SolOpt, SolId, SInfo> for Sp<Obj, Opt>
where
    SolId: Id,
    SInfo: SolInfo,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolOpt: Uncomputed<SolId, Opt, SInfo, Raw = Arc<[Opt::TypeDom]>> + IntoComputed,
    SolOpt::Twin<Obj>:
        Uncomputed<SolId, Obj, SInfo, Twin<Opt> = SolOpt, Raw = Arc<[Obj::TypeDom]>> + IntoComputed,
{
    type SolShape = Pair<SolOpt::Twin<Obj>, SolOpt, SolId, Self::Obj, Self::Opt, SInfo>;

    /// Map an optimizer-side solution into the objective-side space. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_obj(&self, inp: SolOpt) -> Self::SolShape {
        let outx: Vec<LinkTyObj<Self>> = self
            .var
            .iter()
            .zip(inp.get_x().iter())
            .map(|(v, xi)| v.onto_obj(xi).unwrap())
            .collect();
        let solobj = Solution::twin::<Obj>(&inp, outx.into());
        Pair::new(solobj, inp)
    }

    /// Map an objective-side solution into the optimizer-side space. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_opt(&self, inp: SolOpt::Twin<Obj>) -> Self::SolShape {
        let outx: Vec<LinkTyOpt<Self>> = self
            .var
            .iter()
            .zip(inp.get_x().iter())
            .map(|(v, xi)| v.onto_opt(xi).unwrap())
            .collect();
        let solopt = Solution::twin::<Opt>(&inp, outx.into());
        Pair::new(inp, solopt)
    }

    /// Sample a new objective-side solution.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_obj<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt::Twin<Obj> {
        let outx: Vec<_> = self.var.iter().map(|v| v.sample_obj(rng)).collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }

    /// Sample a new optimizer-side solution.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_opt<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt {
        let outx: Vec<_> = self.var.iter().map(|v| v.sample_opt(rng)).collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }

    /// Check whether a solution belongs to the objective domain.
    /// See module-level examples in [`crate::searchspace`].
    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<
                SolId,
                Self::Obj,
                SInfo,
                Raw = <SolOpt::Twin<Obj> as Solution<SolId, Self::Obj, SInfo>>::Raw,
            > + Send
            + Sync,
    {
        inp.get_x()
            .iter()
            .zip(&self.var)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    /// Check whether a solution belongs to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = <SolOpt as Solution<SolId, Opt, SInfo>>::Raw>
            + Send
            + Sync,
    {
        inp.get_x()
            .iter()
            .zip(&self.var)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    /// Map a vector of optimizer-side solutions into the objective space. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_obj(&self, inp: Vec<SolOpt>) -> Vec<Self::SolShape> {
        inp.into_iter().map(|i| self.onto_obj(i)).collect()
    }

    /// Map a vector of objective-side solutions into the optimizer space. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_opt(&self, inp: Vec<SolOpt::Twin<Obj>>) -> Vec<Self::SolShape> {
        inp.into_iter().map(|i| self.onto_opt(i)).collect()
    }

    /// Sample multiple objective-side solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_obj<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Obj>> {
        (0..size)
            .map(|_| {
                <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(self, rng, info.clone())
            })
            .collect()
    }

    /// Sample multiple optimizer-side solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_opt<R: Rng>(&self, rng: &mut R, size: usize, info: Arc<SInfo>) -> Vec<SolOpt> {
        (0..size)
            .map(|_| self.sample_opt(rng, info.clone()))
            .collect()
    }

    /// Check whether all solutions belong to the objective domain.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = RawObj<Self::SolShape, SolId, SInfo>>
            + Send
            + Sync,
    {
        inp.iter().all(|sol| {
            <Sp<Obj, Opt> as Searchspace<SolOpt, SolId, SInfo>>::is_in_obj::<S>(self, sol)
        })
    }

    /// Check whether all solutions belong to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = RawOpt<Self::SolShape, SolId, SInfo>>
            + Send
            + Sync,
    {
        inp.iter().all(|sol| {
            <Sp<Obj, Opt> as Searchspace<SolOpt, SolId, SInfo>>::is_in_opt::<S>(self, sol)
        })
    }

    /// Sample an objective-side solution and map it to the optimizer space.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> Self::SolShape {
        let s = <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(self, rng, info.clone()); // sample
        self.onto_opt(s)
    }

    /// Sample multiple [`SolutionShape`](crate::solution::SolutionShape)
    /// by sampling objective-side solutions then mapping them to the optimizer space.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_pair<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::SolShape> {
        (0..size)
            .map(|_| {
                let s = <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(
                    self,
                    rng,
                    info.clone(),
                ); // sample
                self.onto_opt(s)
            })
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_obj<F, R>(
        &self,
        f: F,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Self::Obj>>
    where
        F: Fn(SolOpt::Twin<Self::Obj>) -> SolOpt::Twin<Self::Obj>,
        R: Rng,
    {
        (0..size)
            .map(|_| {
                f(<Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(
                    self,
                    rng,
                    info.clone(),
                ))
            })
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_opt<F, R>(&self, f: F, rng: &mut R, size: usize, info: Arc<SInfo>) -> Vec<SolOpt>
    where
        F: Fn(SolOpt) -> SolOpt,
        R: Rng,
    {
        (0..size)
            .map(|_| {
                f(<Self as Searchspace<SolOpt, SolId, SInfo>>::sample_opt(
                    self,
                    rng,
                    info.clone(),
                ))
            })
            .collect()
    }

    /// Equivalent to `vec_sample_pair` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_pair<F, R>(
        &self,
        f: F,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::SolShape>
    where
        F: Fn(Self::SolShape) -> Self::SolShape,
        R: Rng,
    {
        (0..size)
            .map(|_| {
                let s = <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_pair(
                    self,
                    rng,
                    info.clone(),
                );
                f(s)
            })
            .collect()
    }
}

/// Sequential [`Searchspace`] implementation for objective-only domains.
impl<SolOpt, SolId, Obj, SInfo> Searchspace<SolOpt, SolId, SInfo> for Sp<Obj, NoDomain>
where
    SolId: Id,
    SInfo: SolInfo,
    Obj: Domain,
    SolOpt: Solution<SolId, Obj, SInfo, Raw = Arc<[Obj::TypeDom]>, Twin<Obj> = SolOpt>
        + Uncomputed<SolId, Obj, SInfo>
        + IntoComputed,
{
    type SolShape = Lone<SolOpt, SolId, Obj, SInfo>;

    /// Wrap an optimizer-side solution directly as a lone solution. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_obj(&self, inp: SolOpt) -> Self::SolShape {
        Lone::new(inp)
    }

    /// Wrap an objective-side solution directly as a lone solution. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_opt(&self, inp: SolOpt::Twin<Obj>) -> Self::SolShape {
        Lone::new(inp)
    }

    /// Sample a new objective-side solution.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_obj<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt::Twin<Obj> {
        let outx: Vec<_> = self.var.iter().map(|v| v.sample_obj(rng)).collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }

    /// Sample a new optimizer-side solution (identical to objective-side when using [`NoDomain`]).
    /// See module-level examples in [`crate::searchspace`].
    fn sample_opt<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt::Twin<Obj> {
        let outx: Vec<_> = self.var.iter().map(|v| v.sample_opt(rng)).collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }

    /// Check whether a solution belongs to the objective domain.
    /// See module-level examples in [`crate::searchspace`].
    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<
                SolId,
                Self::Obj,
                SInfo,
                Raw = <SolOpt::Twin<Obj> as Solution<SolId, Self::Obj, SInfo>>::Raw,
            > + Send
            + Sync,
    {
        inp.get_x()
            .iter()
            .zip(&self.var)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    /// Check whether a solution belongs to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<
                SolId,
                Self::Opt,
                SInfo,
                Raw = <SolOpt::Twin<Obj> as Solution<SolId, Self::Opt, SInfo>>::Raw,
            >,
    {
        inp.get_x()
            .iter()
            .zip(&self.var)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    /// Wrap a vector of optimizer-side solutions as lone solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_obj(&self, inp: Vec<SolOpt>) -> Vec<Self::SolShape> {
        inp.into_iter().map(|sol| Lone::new(sol)).collect()
    }

    /// Wrap a vector of objective-side solutions as lone solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_opt(&self, inp: Vec<SolOpt::Twin<Obj>>) -> Vec<Self::SolShape> {
        inp.into_iter().map(|sol| Lone::new(sol)).collect()
    }

    /// Sample multiple objective-side solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_obj<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Obj>> {
        (0..size)
            .map(|_| {
                <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(self, rng, info.clone())
            })
            .collect()
    }

    /// Sample multiple optimizer-side solutions (same as objective-side in this mode).
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_opt<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Obj>> {
        (0..size)
            .map(|_| self.sample_opt(rng, info.clone()))
            .collect()
    }

    /// Sample an objective-side solution and wrap it as a [`Lone`] [`SolutionShape`](crate::solution::SolutionShape).
    /// See module-level examples in [`crate::searchspace`].
    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> Self::SolShape {
        let s = <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(self, rng, info.clone()); // sample
        Lone::new(s)
    }

    /// Sample multiple [`Lone`] [`SolutionShape`](crate::solution::SolutionShape).
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_pair<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::SolShape> {
        (0..size)
            .map(|_| {
                let s = <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(
                    self,
                    rng,
                    info.clone(),
                ); // sample
                Lone::new(s)
            })
            .collect()
    }

    /// Check whether all solutions belong to the objective domain.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync,
    {
        inp.iter().all(|sol| {
            <Sp<Obj, NoDomain> as Searchspace<SolOpt::Twin<Obj>, SolId, SInfo>>::is_in_obj::<S>(
                self, sol,
            )
        })
    }

    /// Check whether all solutions belong to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync,
    {
        inp.iter().all(|sol| {
            <Sp<Obj, NoDomain> as Searchspace<SolOpt::Twin<Obj>, SolId, SInfo>>::is_in_obj::<S>(
                self, sol,
            )
        })
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_obj<F, R>(
        &self,
        f: F,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Self::Obj>>
    where
        F: Fn(SolOpt::Twin<Self::Obj>) -> SolOpt::Twin<Self::Obj>,
        R: Rng,
    {
        (0..size)
            .map(|_| {
                f(<Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(
                    self,
                    rng,
                    info.clone(),
                ))
            })
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_opt<F, R>(&self, f: F, rng: &mut R, size: usize, info: Arc<SInfo>) -> Vec<SolOpt>
    where
        F: Fn(SolOpt) -> SolOpt,
        R: Rng,
    {
        (0..size)
            .map(|_| {
                f(<Self as Searchspace<SolOpt, SolId, SInfo>>::sample_opt(
                    self,
                    rng,
                    info.clone(),
                ))
            })
            .collect()
    }

    /// Equivalent to `vec_sample_pair` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_pair<F, R>(
        &self,
        f: F,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::SolShape>
    where
        F: Fn(Self::SolShape) -> Self::SolShape,
        R: Rng,
    {
        (0..size)
            .map(|_| {
                let s = <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_pair(
                    self,
                    rng,
                    info.clone(),
                );
                f(s)
            })
            .collect()
    }
}

/// Convert a multi-threaded (via [`rayon`]) searchspace into its sequential counterpart.
impl<Obj: Domain, Opt: Domain> From<SpPar<Obj, Opt>> for Sp<Obj, Opt> {
    fn from(value: SpPar<Obj, Opt>) -> Self {
        Sp { var: value.var }
    }
}

/// CSV support for objective and optimizer sides of the searchspace.
impl<Obj, Opt> CSVLeftRight<Sp<Obj, Opt>, Arc<[LinkTyObj<Self>]>, Arc<[LinkTyOpt<Self>]>>
    for Sp<Obj, Opt>
where
    Obj: OntoDom<Opt> + CSVWritable<(), LinkTyObj<Self>>,
    Opt: OntoDom<Obj> + CSVWritable<(), LinkTyOpt<Self>>,
    Var<Obj, Opt>: CSVLeftRight<Var<Obj, Opt>, LinkTyObj<Self>, LinkTyOpt<Self>>,
{
    /// Build the CSV header by concatenating variable headers.
    /// See module-level examples in [`crate::searchspace`].
    fn header(elem: &Sp<Obj, Opt>) -> Vec<String> {
        elem.var.iter().flat_map(Var::<Obj, Opt>::header).collect()
    }

    /// Write objective-side values as CSV columns.
    /// See module-level examples in [`crate::searchspace`].
    fn write_left(&self, comp: &Arc<[LinkTyObj<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_left(c))
            .collect()
    }

    /// Write optimizer-side values as CSV columns.
    /// See module-level examples in [`crate::searchspace`].
    fn write_right(&self, comp: &Arc<[LinkTyOpt<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_right(c))
            .collect()
    }
}

/// CSV support for objective-only searchspaces.
impl<Obj> CSVLeftRight<Sp<Obj, NoDomain>, Arc<[LinkTyObj<Self>]>, Arc<[LinkTyOpt<Self>]>>
    for Sp<Obj, NoDomain>
where
    Obj: Domain + CSVWritable<(), LinkTyObj<Self>>,
    Var<Obj, NoDomain>: CSVLeftRight<Var<Obj, NoDomain>, LinkTyObj<Self>, LinkTyOpt<Self>>,
{
    /// Build the CSV header by concatenating variable headers.
    /// See module-level examples in [`crate::searchspace`].
    fn header(elem: &Sp<Obj, NoDomain>) -> Vec<String> {
        elem.var
            .iter()
            .flat_map(Var::<Obj, NoDomain>::header)
            .collect()
    }

    /// Write objective-side values as CSV columns.
    /// See module-level examples in [`crate::searchspace`].
    fn write_left(&self, comp: &Arc<[LinkTyObj<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_left(c))
            .collect()
    }

    /// Write optimizer-side values as CSV columns (same as objective-side in this mode).
    /// See module-level examples in [`crate::searchspace`].
    fn write_right(&self, comp: &Arc<[LinkTyOpt<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_right(c))
            .collect()
    }
}
