//! A multi-threaded [`Searchspace`] implementation backed by a boxed slice of [`Var`]s, based on [`rayon`].
//!
//! The [`SpPar`] type provides a multi-threaded [`Searchspace`] built on [`rayon`]. It mirrors the
//! sequential version in [`Sp`] and can be converted to or from it.
//!
//! For end-to-end usage (including the [`hpo!`](../../../tantale/macros/macro.hpo.html) and [`objective!`](../../../tantale/macros/macro.objective.html)
//! macros), see the module-level examples in [`crate::searchspace`].
//!
//! # Example
//! ```
//! use tantale::core::{SpPar, Var, Real, Unit, Uniform};
//!
//! let v1 = Var::<Real, Unit>::new("x", Real::new(0.0, 1.0, Uniform), Unit::new(Uniform));
//! let v2 = Var::<Real, Unit>::new("y", Real::new(-1.0, 1.0, Uniform), Unit::new(Uniform));
//! let sp = SpPar { var: vec![v1, v2].into_boxed_slice() };
//!
//! assert_eq!(sp.var.len(), 2);
//! ```
//!
//! # Example: convert from [`Sp`]
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
//! use tantale::core::{BaseSol, EmptyInfo, HasId, SId, Searchspace, SpPar, Var, Real, Unit, Uniform};
//! use std::sync::Arc;
//!
//! let v1 = Var::<Real, Unit>::new("x", Real::new(0.0, 1.0, Uniform), Unit::new(Uniform));
//! let v2 = Var::<Real, Unit>::new("y", Real::new(-1.0, 1.0, Uniform), Unit::new(Uniform));
//! let sp = SpPar { var: vec![v1, v2].into_boxed_slice() };
//! let info = Arc::new(EmptyInfo {});
//! let mut rng = rand::rng();
//!
//! type OptSol = BaseSol<SId, Unit, EmptyInfo>;
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
//! let _ = pair.id();
//! ```

use crate::{
    FuncWrapper, HasId, HasInfo, HasVariables, HasX, Sp, aliases::{GetRaw, GetSInfo, GetSObj, GetSOpt}, domain::{
        Domain, NoDomain, PreDomain, onto::{Linked, OntoDom},
    }, searchspace::Searchspace, solution::{Lone, Pair, Uncomputed}, variable::{Var, var::ToName},
};

use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use std::sync::Arc;

use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};
use rayon::prelude::*;

/// A basic parallel [`Searchspace`] made of a boxed slice of [`Var`].
pub struct SpPar<Obj: Domain, Opt: PreDomain> {
    /// Variables that define the [`Objective`](crate::Objective) and [`Optimizer`](crate::Optimizer) domains.
    pub var: Box<[Var<Obj, Opt>]>,
}

impl<Obj: Domain, Opt: PreDomain> SpPar<Obj, Opt> {
    /// Get the number of variables in the searchspace.
    pub fn size(&self) -> usize {
        self.var.len()
    }
}

/// Link objective and optimizer domains when both sides are present.
impl<Obj: OntoDom<Opt>, Opt: OntoDom<Obj>> Linked for SpPar<Obj, Opt> {
    type Obj = Obj;
    type Opt = Opt;
    type TrueOpt = Opt;
}

/// Link objective and optimizer domains (objective domains with itself) when using [`NoDomain`].
impl<Obj: Domain> Linked for SpPar<Obj, NoDomain> {
    type Obj = Obj;
    type Opt = Obj;
    type TrueOpt = NoDomain;
}

impl<Obj: OntoDom<Opt>, Opt: OntoDom<Obj>> HasVariables for SpPar<Obj, Opt> {
    fn obj_at(&self, index: usize) -> Option<&Self::Obj> {
        self.var.get(index).map(|v| v.domain_obj.as_ref())
    }

    fn opt_at(&self, index: usize) -> Option<&Self::Opt> {
        self.var.get(index).map(|v| v.domain_opt.as_ref())
    }

    fn size(&self) -> usize {
        self.var.len()
    }

    fn names(&self) -> Vec<String> {
        self.var.iter().map(|v|v.to_name()).collect()
    }
    
    fn get_vars(&self) -> &[Var<Self::Obj, Self::TrueOpt>] {
        &self.var
    }
}

impl<Obj: Domain> HasVariables for SpPar<Obj, NoDomain> {
    fn obj_at(&self, index: usize) -> Option<&Self::Obj> {
        self.var.get(index).map(|v| v.domain_obj.as_ref())
    }

    fn opt_at(&self, index: usize) -> Option<&Self::Opt> {
        self.var.get(index).map(|v| v.domain_obj.as_ref())
    }

    fn size(&self) -> usize {
        self.var.len()
    }

    fn names(&self) -> Vec<String> {
        self.var.iter().map(|v|v.to_name()).collect()
    }

    fn get_vars(&self) -> &[Var<Self::Obj, Self::TrueOpt>] {
        &self.var
    }
}

/// Parallel [`Searchspace`] implementation for paired objective/optimizer domains.
impl<F, Obj, Opt> Searchspace<F> for SpPar<Obj, Opt>
where
    F: FuncWrapper<Dom = Obj>,
    F::Sol<Obj>: Uncomputed<X = Arc<[Obj::Element]>> + Send + Sync,
    F::Sol<Opt>: Uncomputed<X = Arc<[Opt::Element]>>
    + HasId<Id = <F::Sol<Obj> as HasId>::Id>
    + HasInfo<Info = <F::Sol<Obj> as HasInfo>::Info>
    + Send + Sync,
    <F::Sol<Obj> as HasInfo>::Info: Send + Sync,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    Obj::Element: Send + Sync,
    Opt::Element: Send + Sync,
{
    type Shape = Pair<F::Sol<Obj>, F::Sol<Opt>>;

    /// Map an optimizer-side solution into the objective-side space. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_obj(&self, inp: F::Sol<Opt>) -> Self::Shape {
        let var_it = self.var.par_iter();
        let outx: Vec<Obj::Element> = inp
            .ref_x()
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_obj(i).unwrap())
            .collect();
        let solobj = GetSObj::<Self::Shape>::new(inp.id(), outx, inp.sinfo());
        Pair::new(solobj, inp)
    }

    /// Map an objective-side solution into the optimizer-side space. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_opt(&self, inp: F::Sol<Obj>) -> Self::Shape {
        let var_it = self.var.par_iter();
        let outx: Vec<Opt::Element> = inp
            .ref_x()
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_opt(i).unwrap())
            .collect();
        let solopt = GetSOpt::<Self::Shape>::new(inp.id(), outx, inp.sinfo());
        Pair::new(inp, solopt)
    }

    fn sample_raw_obj<R: Rng>(&self, rng: &mut R) -> GetRaw<GetSObj<Self::Shape>> {
        let seeds: Vec<u64> = self.var.iter().map(|_| rng.random()).collect();
        let variter = self.var.par_iter();
        // Generate seeds for each thread
        seeds.into_par_iter()
            .zip(variter)
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, (seed, var)| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    var.sample_obj(thread_rng)
                },
            )
            .collect()
    }

    fn sample_raw_opt<R: Rng>(&self, rng: &mut R) -> GetRaw<GetSOpt<Self::Shape>> {
        let seeds: Vec<u64> = self.var.iter().map(|_| rng.random()).collect();
        let variter = self.var.par_iter();
        // Generate seeds for each thread
        seeds.into_par_iter()
            .zip(variter)
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, (seed, var)| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    var.sample_opt(thread_rng)
                },
            )
            .collect()
    }

    /// Check whether a solution belongs to the objective domain.
    /// See module-level examples in [`crate::searchspace`].
    fn contains_obj(&self, inp: &GetRaw<GetSObj<Self::Shape>>) -> bool
    {
        let variter = self.var.par_iter();
        inp.par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    /// Check whether a solution belongs to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn contains_opt(&self, inp: &GetRaw<GetSOpt<Self::Shape>>) -> bool
    {
        let variter = self.var.par_iter();
        inp.par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    /// Map a vector of optimizer-side solutions into the objective space.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_obj(&self, inp: Vec<GetSOpt<Self::Shape>>) -> Vec<Self::Shape> {
        inp.into_par_iter().map(|sol| <SpPar<Obj, Opt> as Searchspace<F>>::onto_obj(self, sol)).collect()
    }

    /// Map a vector of objective-side solutions into the optimizer space.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_opt(&self, inp: Vec<GetSObj<Self::Shape>>) -> Vec<Self::Shape> {
        inp.into_par_iter().map(|sol| <SpPar<Obj, Opt> as Searchspace<F>>::onto_opt(self, sol)).collect()
    }

    /// Sample multiple objective-side solutions using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_obj<R: Rng>(&self,rng: &mut R, size: usize, info: Arc<F::Info>) -> Vec<GetSObj<Self::Shape>> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <SpPar<Obj, Opt> as Searchspace<F>>::sample_obj::<rand::prelude::StdRng>(self, thread_rng, info.clone())
                },
            )
            .collect()
    }

    fn vec_sample_raw_obj<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<GetRaw<GetSObj<Self::Shape>>> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <SpPar<Obj, Opt> as Searchspace<F>>::sample_raw_obj::<rand::prelude::StdRng>(self, thread_rng)
                },
            )
            .collect()
    }

    /// Sample multiple optimizer-side solutions using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_opt<R: Rng>(&self, rng: &mut R, size: usize, info: Arc<F::Info>) -> Vec<GetSOpt<Self::Shape>> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <SpPar<Obj, Opt> as Searchspace<F>>::sample_opt::<StdRng>(self, thread_rng, info.clone())
                },
            )
            .collect()
    }

    fn vec_sample_raw_opt<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<GetRaw<GetSOpt<Self::Shape>>> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <SpPar<Obj, Opt> as Searchspace<F>>::sample_raw_opt::<StdRng>(self, thread_rng)
                },
            )
            .collect()
    }

    /// Sample an objective-side solution and map it to the optimizer space.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<F::Info>) -> Self::Shape {
        <SpPar<Obj, Opt> as Searchspace<F>>::onto_opt(self, <SpPar<Obj, Opt> as Searchspace<F>>::sample_obj::<R>(self, rng, info.clone()))
    }

    /// Sample multiple pairs by sampling objective-side solutions then mapping them.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_pair<R: Rng>(&self,rng: &mut R,size: usize,info: Arc<F::Info>) -> Vec<Self::Shape> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <SpPar<Obj, Opt> as Searchspace<F>>::sample_pair::<rand::prelude::StdRng>(self, thread_rng, info.clone())
                },
            )
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_obj<A, R>(&self,f: A,rng: &mut R,size: usize,info: Arc<F::Info>) -> Vec<GetSObj<Self::Shape>>
    where
        A: Fn(GetSObj<Self::Shape>) -> GetSObj<Self::Shape> + Send + Sync,
        R: Rng,
    {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    f(<SpPar<Obj, Opt> as Searchspace<F>>::sample_obj::<StdRng>(self, thread_rng, info.clone()))
                },
            )
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_opt<A, R>(&self, f: A, rng: &mut R, size: usize, info: Arc<F::Info>) -> Vec<GetSOpt<Self::Shape>>
    where
        A: Fn(GetSOpt<Self::Shape>) -> GetSOpt<Self::Shape> + Send + Sync,
        R: Rng,
    {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    f(<SpPar<Obj, Opt> as Searchspace<F>>::sample_opt::<StdRng>(self, thread_rng, info.clone()))
                },
            )
            .collect()
    }
}

/// Parallel [`Searchspace`] implementation for objective-only domains.
impl<F, Dom> Searchspace<F> for SpPar<Dom, NoDomain>
where
    F: FuncWrapper<Dom = Dom>,
    F::Sol<Dom>: Uncomputed<X = Arc<[Dom::Element]>> + Send + Sync,
    <F::Sol<Dom> as HasInfo>::Info: Send + Sync,
    Dom: Domain + Send + Sync,
    Dom::Element: Send + Sync,
{
    /// Wrap an objective-side solution directly as a lone solution.
    type Shape = Lone<F::Sol<Dom>>;

    /// Wrap an optimizer-side solution directly as a lone solution. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_obj(&self, inp: F::Sol<Dom>) -> Self::Shape {
        Lone::new(inp)
    }

    /// Wrap an objective-side solution directly as a lone solution. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_opt(&self, inp: F::Sol<Dom>) -> Self::Shape {
        Lone::new(inp)
    }

    fn sample_raw_obj<R: Rng>(&self, rng: &mut R) -> GetRaw<F::Sol<Dom>> {
        let seeds: Vec<u64> = self.var.iter().map(|_| rng.random()).collect();
        let variter = self.var.par_iter();
        // Generate seeds for each thread
        seeds.into_par_iter()
            .zip(variter)
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, (seed, var)| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    var.sample_obj(thread_rng)
                },
            )
            .collect()
    }

    fn sample_raw_opt<R: Rng>(&self, rng: &mut R) -> GetRaw<F::Sol<Dom>> {
        <SpPar<Dom, NoDomain> as Searchspace<F>>::sample_raw_obj::<R>(self, rng)
    }

    /// Check whether a solution belongs to the objective domain.
    /// See module-level examples in [`crate::searchspace`].
    fn contains_obj(&self, inp: &GetRaw<F::Sol<Dom>>) -> bool {
        let variter = self.var.par_iter();
        inp.par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    /// Check whether a solution belongs to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn contains_opt(&self, inp: &GetRaw<F::Sol<Dom>>) -> bool {
        let variter = self.var.par_iter();
        inp.par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    /// Wrap a vector of objective-side solutions as lone solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_obj(&self, inp: Vec<F::Sol<Dom>>) -> Vec<Self::Shape> {
        inp.into_par_iter().map(Lone::new).collect()
    }

    /// Wrap a vector of objective-side solutions as lone solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_opt(&self, inp: Vec<F::Sol<Dom>>) -> Vec<Self::Shape> {
        inp.into_par_iter().map(Lone::new).collect()
    }

    /// Sample multiple objective-side solutions using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_obj<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<GetSInfo<F::Sol<Dom>>>,
    ) -> Vec<F::Sol<Dom>> {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <SpPar<Dom, NoDomain> as Searchspace<F>>::sample_obj::<StdRng>(self, thread_rng, info.clone())
                },
            )
            .collect()
    }

    fn vec_sample_raw_obj<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<GetRaw<F::Sol<Dom>>> {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <SpPar<Dom, NoDomain> as Searchspace<F>>::sample_raw_obj::<rand::prelude::StdRng>(self, thread_rng)
                },
            )
            .collect()
    }

    /// Sample multiple optimizer-side solutions using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_opt<R: Rng>(&self,rng: &mut R,size: usize,info: Arc<GetSInfo<F::Sol<Dom>>>) -> Vec<F::Sol<Dom>> {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <SpPar<Dom, NoDomain> as Searchspace<F>>::sample_opt::<StdRng>(self, thread_rng, info.clone())
                },
            )
            .collect()
    }

    fn vec_sample_raw_opt<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<GetRaw<F::Sol<Dom>>> {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <SpPar<Dom, NoDomain> as Searchspace<F>>::sample_raw_opt::<StdRng>(self, thread_rng)
                },
            )
            .collect()
    }

    /// Sample an objective-side solution and wrap it as a lone solution.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<GetSInfo<F::Sol<Dom>>>) -> Self::Shape {
        Lone::new(<SpPar<Dom, NoDomain> as Searchspace<F>>::sample_opt::<R>(self, rng, info))
    }

    /// Sample multiple lone solutions using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_pair<R: Rng>(&self,rng: &mut R,size: usize,info: Arc<GetSInfo<F::Sol<Dom>>>) -> Vec<Self::Shape> {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <SpPar<Dom, NoDomain> as Searchspace<F>>::sample_pair::<StdRng>(self, thread_rng, info.clone())
                },
            )
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_obj<A, R>(&self,f: A,rng: &mut R,size: usize,info: Arc<GetSInfo<F::Sol<Dom>>>) -> Vec<F::Sol<Dom>>
    where
        A: Fn(F::Sol<Dom>) -> F::Sol<Dom> + Send + Sync,
        R: Rng,
    {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    f(<SpPar<Dom, NoDomain> as Searchspace<F>>::sample_obj::<StdRng>(self, thread_rng, info.clone()))
                },
            )
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_opt<A, R>(&self, f: A, rng: &mut R, size: usize, info: Arc<GetSInfo<F::Sol<Dom>>>) -> Vec<F::Sol<Dom>>
    where
        A: Fn(F::Sol<Dom>) -> F::Sol<Dom> + Send + Sync,
        R: Rng,
    {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    f(<SpPar<Dom, NoDomain> as Searchspace<F>>::sample_opt::<StdRng>(self, thread_rng, info.clone()))
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