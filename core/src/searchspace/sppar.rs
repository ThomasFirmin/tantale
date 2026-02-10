//! A multi-threaded [`Searchspace`] implementation backed by a boxed slice of [`Var`]s, based on [`rayon`].
//!
//! The [`SpPar`] type provides a multi-threaded [`Searchspace`] built on [`rayon`]. It mirrors the
//! sequential version in [`Sp`](crate::Sp) and can be converted to or from it.
//!
//! For end-to-end usage (including the [`hpo!`](../../../tantale/macros/macro.hpo.html) and [`objective!`](../../../tantale/macros/macro.objective.html)
//! macros), see the module-level examples in [`crate::searchspace`].
//!
//! # Example
//! ```
//! use tantale::core::{SpPar, Var, Real, Unit, Uniform};
//!
//! let v1 = Var::new("x", Real::new(0.0, 1.0, Uniform), Unit::new(Uniform));
//! let v2 = Var::new("y", Real::new(-1.0, 1.0, Uniform), Unit::new(Uniform));
//! let sp = SpPar { var: vec![v1, v2].into_boxed_slice() };
//!
//! assert_eq!(sp.var.len(), 2);
//! ```
//!
//! # Example: convert from [`Sp`](crate::Sp)
//! ```
//! use tantale::core::{Sp, SpPar, Var, Real, Unit, Uniform};
//!
//! let v1 = Var::new("x", Real::new(0.0, 1.0, Uniform), Unit::new(Uniform));
//! let v2 = Var::new("y", Real::new(-1.0, 1.0, Uniform), Unit::new(Uniform));
//! let sp = Sp { var: vec![v1, v2].into_boxed_slice() };
//! let sp_par: SpPar<_, _> = sp.into();
//!
//! assert_eq!(sp_par.var.len(), 2);
//! ```
//!
//! # Example: sampling pairs with [`Searchspace`]
//! ```
//! use tantale::core::{BasePartial, EmptyInfo, SId, Searchspace, SpPar, Var, Real, Unit, Uniform};
//! use std::sync::Arc;
//!
//! let v1 = Var::new("x", Real::new(0.0, 1.0, Uniform), Unit::new(Uniform));
//! let v2 = Var::new("y", Real::new(-1.0, 1.0, Uniform), Unit::new(Uniform));
//! let sp = SpPar { var: vec![v1, v2].into_boxed_slice() };
//! let info = Arc::new(EmptyInfo {});
//! let mut rng = rand::rng();
//!
//! type OptSol = BasePartial<SId, Unit, EmptyInfo>;
//! let pair = <SpPar<Real, Unit> as Searchspace<OptSol, SId, EmptyInfo>>::sample_pair(
//!     &sp,
//!     &mut rng,
//!     info.clone(),
//! );
//! let pairs = <SpPar<Real, Unit> as Searchspace<OptSol, SId, EmptyInfo>>::vec_sample_pair(
//!     &sp,
//!     &mut rng,
//!     3,
//!     info.clone(),
//! );
//! let obj = <SpPar<Real, Unit> as Searchspace<OptSol, SId, EmptyInfo>>::sample_obj(
//!     &sp,
//!     &mut rng,
//!     info,
//! );
//! let _pair_from_obj = <SpPar<Real, Unit> as Searchspace<OptSol, SId, EmptyInfo>>::onto_opt(
//!     &sp,
//!     obj,
//! );
//!
//! assert_eq!(pairs.len(), 3);
//! let _ = pair.get_id();
//! ```

use crate::{
    Sp,
    domain::{
        Domain, NoDomain, PreDomain,
        onto::{LinkTyObj, LinkTyOpt, Linked, OntoDom},
    },
    recorder::csv::{CSVLeftRight, CSVWritable},
    searchspace::{Searchspace, SolInfo},
    solution::{Id, IntoComputed, Lone, Pair, Solution, Uncomputed},
    variable::Var,
};

use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};
use rayon::prelude::*;

/// A basic parallel [`Searchspace`] made of a boxed slice of [`Var`].
pub struct SpPar<Obj: Domain, Opt: PreDomain> {
    /// Variables that define the [`Objective`](crate::Objective) and [`Optimizer`](crate::Optimizer) domains.
    pub var: Box<[Var<Obj, Opt>]>,
}

/// Link objective and optimizer domains when both sides are present.
impl<Obj: OntoDom<Opt>, Opt: OntoDom<Obj>> Linked for SpPar<Obj, Opt> {
    type Obj = Obj;
    type Opt = Opt;
}

/// Link objective and optimizer domains (objective domains with itself) when using [`NoDomain`].
impl<Obj: Domain> Linked for SpPar<Obj, NoDomain> {
    type Obj = Obj;
    type Opt = Obj;
}

/// Parallel [`Searchspace`] implementation for paired objective/optimizer domains.
impl<SolOpt, SolId, Obj, Opt, SInfo> Searchspace<SolOpt, SolId, SInfo> for SpPar<Obj, Opt>
where
    SolId: Id + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    SolOpt: Uncomputed<SolId, Opt, SInfo, Raw = Arc<[Opt::TypeDom]>> + IntoComputed + Send + Sync,
    SolOpt::Twin<Obj>: Uncomputed<SolId, Obj, SInfo, Twin<Opt> = SolOpt, Raw = Arc<[Obj::TypeDom]>>
        + IntoComputed
        + Send
        + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
{
    type SolShape = Pair<SolOpt::Twin<Obj>, SolOpt, SolId, Self::Obj, Self::Opt, SInfo>;

    /// Map an optimizer-side solution into the objective-side space. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_obj(&self, inp: SolOpt) -> Self::SolShape {
        let var_it = self.var.par_iter();
        let outx: Vec<LinkTyObj<Self>> = inp
            .get_x()
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_obj(i).unwrap())
            .collect();
        let solobj = Solution::twin::<Obj>(&inp, outx.into());
        Pair::new(solobj, inp)
    }

    /// Map an objective-side solution into the optimizer-side space. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_opt(&self, inp: SolOpt::Twin<Obj>) -> Self::SolShape {
        let var_it = self.var.par_iter();
        let outx: Vec<LinkTyOpt<Self>> = inp
            .get_x()
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_opt(i).unwrap())
            .collect();
        let solopt = Solution::twin::<Opt>(&inp, outx.into());
        Pair::new(inp, solopt)
    }

    /// Sample a new objective-side solution using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_obj<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt::Twin<Obj> {
        let seeds: Vec<u64> = self.var.iter().map(|_| rng.random()).collect();
        let variter = self.var.par_iter();
        // Generate seeds for each thread
        let outx: Vec<_> = seeds
            .into_par_iter()
            .zip(variter)
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, (seed, var)| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    var.sample_obj(thread_rng)
                },
            )
            .collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }

    /// Sample a new optimizer-side solution using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_opt<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt {
        let seeds: Vec<u64> = self.var.iter().map(|_| rng.random()).collect();
        let variter = self.var.par_iter();
        // Generate seeds for each thread
        let outx: Vec<_> = seeds
            .into_par_iter()
            .zip(variter)
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, (seed, var)| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    var.sample_opt(thread_rng)
                },
            )
            .collect();
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
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
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
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    /// Map a vector of optimizer-side solutions into the objective space.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_obj(&self, inp: Vec<SolOpt>) -> Vec<Self::SolShape> {
        inp.into_par_iter().map(|sol| self.onto_obj(sol)).collect()
    }

    /// Map a vector of objective-side solutions into the optimizer space.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_opt(&self, inp: Vec<SolOpt::Twin<Obj>>) -> Vec<Self::SolShape> {
        inp.into_par_iter().map(|sol| self.onto_opt(sol)).collect()
    }

    /// Sample multiple objective-side solutions using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_obj<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Obj>> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(
                        self,
                        thread_rng,
                        info.clone(),
                    )
                },
            )
            .collect()
    }

    /// Sample multiple optimizer-side solutions using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_opt<R: Rng>(&self, rng: &mut R, size: usize, info: Arc<SInfo>) -> Vec<SolOpt> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_opt(
                        self,
                        thread_rng,
                        info.clone(),
                    )
                },
            )
            .collect()
    }

    /// Check whether all solutions belong to the objective domain.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync,
    {
        inp.par_iter()
            .all(|sol| <Self as Searchspace<SolOpt, SolId, SInfo>>::is_in_obj::<S>(self, sol))
    }

    /// Check whether all solutions belong to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = Arc<[Opt::TypeDom]>> + Send + Sync,
    {
        inp.par_iter()
            .all(|sol| <Self as Searchspace<SolOpt, SolId, SInfo>>::is_in_opt::<S>(self, sol))
    }

    /// Sample an objective-side solution and map it to the optimizer space.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> Self::SolShape {
        let s = <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(self, rng, info.clone()); // sample
        self.onto_opt(s)
    }

    /// Sample multiple pairs by sampling objective-side solutions then mapping them.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_pair<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::SolShape> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_pair(
                        self,
                        thread_rng,
                        info.clone(),
                    )
                },
            )
            .collect()
    }
}

/// Parallel [`Searchspace`] implementation for objective-only domains.
impl<SolOpt, SolId, Obj, SInfo> Searchspace<SolOpt, SolId, SInfo> for SpPar<Obj, NoDomain>
where
    SolId: Id + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Obj: Domain + Send + Sync,
    SolOpt: Solution<SolId, Obj, SInfo, Raw = Arc<[Obj::TypeDom]>, Twin<Obj> = SolOpt>
        + Uncomputed<SolId, Obj, SInfo>
        + IntoComputed
        + Send
        + Sync,
    Obj::TypeDom: Send + Sync,
{
    type SolShape = Lone<SolOpt, SolId, Obj, SInfo>;

    /// Wrap an optimizer-side solution directly as a lone solution. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_obj(&self, inp: SolOpt::Twin<Obj>) -> Self::SolShape {
        Lone::new(inp)
    }

    /// Wrap an objective-side solution directly as a lone solution. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_opt(&self, inp: SolOpt::Twin<Obj>) -> Self::SolShape {
        Lone::new(inp)
    }

    /// Sample a new objective-side solution using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_obj<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt::Twin<Obj> {
        let seeds: Vec<u64> = self.var.iter().map(|_| rng.random()).collect();
        let variter = self.var.par_iter();
        // Generate seeds for each thread
        let outx: Vec<_> = seeds
            .into_par_iter()
            .zip(variter)
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, (seed, var)| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    var.sample_obj(thread_rng)
                },
            )
            .collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }

    /// Sample a new optimizer-side solution using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_opt<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt::Twin<Obj> {
        let seeds: Vec<u64> = self.var.iter().map(|_| rng.random()).collect();
        let variter = self.var.par_iter();
        // Generate seeds for each thread
        let outx: Vec<_> = seeds
            .into_par_iter()
            .zip(variter)
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, (seed, var)| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    var.sample_opt(thread_rng)
                },
            )
            .collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }

    /// Check whether a solution belongs to the objective domain.
    /// See module-level examples in [`crate::searchspace`].
    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync,
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    /// Check whether a solution belongs to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = Arc<[Obj::TypeDom]>>,
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    /// Wrap a vector of objective-side solutions as lone solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_obj(&self, inp: Vec<SolOpt::Twin<Obj>>) -> Vec<Self::SolShape> {
        inp.into_par_iter().map(|sol| Lone::new(sol)).collect()
    }

    /// Wrap a vector of objective-side solutions as lone solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_opt(&self, inp: Vec<SolOpt::Twin<Obj>>) -> Vec<Self::SolShape> {
        inp.into_par_iter().map(|sol| Lone::new(sol)).collect()
    }

    /// Sample multiple objective-side solutions using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_obj<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Obj>> {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(
                        self,
                        thread_rng,
                        info.clone(),
                    )
                },
            )
            .collect()
    }

    /// Sample multiple optimizer-side solutions using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_opt<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Obj>> {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_opt(
                        self,
                        thread_rng,
                        info.clone(),
                    )
                },
            )
            .collect()
    }

    /// Check whether all solutions belong to the objective domain.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync,
    {
        inp.par_iter()
            .all(|sol| <Self as Searchspace<SolOpt, SolId, SInfo>>::is_in_obj::<S>(self, sol))
    }

    /// Check whether all solutions belong to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync,
    {
        inp.par_iter()
            .all(|sol| <Self as Searchspace<SolOpt, SolId, SInfo>>::is_in_opt::<S>(self, sol))
    }

    /// Sample an objective-side solution and wrap it as a lone solution.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> Self::SolShape {
        Lone::new(self.sample_opt(rng, info))
    }

    /// Sample multiple lone solutions using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_pair<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::SolShape> {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_pair(
                        self,
                        thread_rng,
                        info.clone(),
                    )
                },
            )
            .collect()
    }
}

/// Convert a sequential searchspace into its parallel counterpart.
impl<Obj: Domain, Opt: Domain> From<Sp<Obj, Opt>> for SpPar<Obj, Opt> {
    fn from(value: Sp<Obj, Opt>) -> Self {
        SpPar { var: value.var }
    }
}

/// CSV support for objective and optimizer sides of the parallel searchspace.
impl<Obj, Opt> CSVLeftRight<SpPar<Obj, Opt>, Arc<[LinkTyObj<Self>]>, Arc<[LinkTyOpt<Self>]>>
    for SpPar<Obj, Opt>
where
    Obj: OntoDom<Opt> + CSVWritable<(), LinkTyObj<Self>>,
    Opt: OntoDom<Obj> + CSVWritable<(), LinkTyOpt<Self>>,
    Var<Obj, Opt>: CSVLeftRight<Var<Obj, Opt>, LinkTyObj<Self>, LinkTyOpt<Self>>,
{
    /// Build the CSV header by concatenating variable headers.
    /// See module-level examples in [`crate::searchspace`].
    fn header(elem: &SpPar<Obj, Opt>) -> Vec<String> {
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

/// CSV support for objective-only parallel searchspaces.
impl<Obj> CSVLeftRight<SpPar<Obj, NoDomain>, Arc<[LinkTyObj<Self>]>, Arc<[LinkTyOpt<Self>]>>
    for SpPar<Obj, NoDomain>
where
    Obj: Domain + CSVWritable<(), LinkTyObj<Self>>,
    Var<Obj, NoDomain>: CSVLeftRight<Var<Obj, NoDomain>, LinkTyObj<Self>, LinkTyOpt<Self>>,
{
    /// Build the CSV header by concatenating variable headers.
    /// See module-level examples in [`crate::searchspace`].
    fn header(elem: &SpPar<Obj, NoDomain>) -> Vec<String> {
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
