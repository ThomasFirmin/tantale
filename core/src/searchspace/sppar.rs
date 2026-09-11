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
    FuncWrapper, HasId, HasInfo, HasVariables, HasX, Problem, Sp, aliases::{GetRaw, GetSInfo, GetSObj, GetSOpt}, domain::{
        Domain, NoDomain, PreDomain, onto::{Linked, OntoDom},
    }, searchspace::{Searchspace, SolInfo}, solution::{Id, Lone, Pair, Solution, Uncomputed}, variable::Var,
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
        self.var.iter().map(
            |v|
            match v.name.1 {
                Some(idx) => format!("{}_{}", v.name.0, idx),
                None => v.name.0.to_string(),
            }
        ).collect()
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
        self.var.iter().map(
            |v|
            match v.name.1 {
                Some(idx) => format!("{}_{}", v.name.0, idx),
                None => v.name.0.to_string(),
            }
        ).collect()
    }
}

/// Parallel [`Searchspace`] implementation for paired objective/optimizer domains.
impl<F, Obj, Opt> Searchspace<F> for SpPar<Obj, Opt>
where
    F: FuncWrapper<Dom = Obj>,
    F::Sol<Obj>: Uncomputed<X = Arc<[Obj::Element]>>,
    F::Sol<Opt>: Uncomputed<X = Arc<[Opt::Element]>>
    + HasId<Id = <F::Sol<Obj> as HasId>::Id>
    + HasInfo<Info = <F::Sol<Obj> as HasInfo>::Info>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
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
        inp.ref_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    /// Check whether a solution belongs to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn contains_opt(&self, inp: &GetRaw<GetSOpt<Self::Shape>>) -> bool
    {
        let variter = self.var.par_iter();
        inp.ref_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    /// Map a vector of optimizer-side solutions into the objective space.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_obj(&self, inp: Vec<GetSOpt<Self::Shape>>) -> Vec<Self::Shape> {
        inp.into_par_iter().map(|sol| self.onto_obj(sol)).collect()
    }

    /// Map a vector of objective-side solutions into the optimizer space.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_opt(&self, inp: Vec<GetSObj<Self::Shape>>) -> Vec<Self::Shape> {
        inp.into_par_iter().map(|sol| self.onto_opt(sol)).collect()
    }

    /// Sample multiple objective-side solutions using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_obj<R: Rng>(&self,rng: &mut R, size: usize, info: Arc<GetSInfo<Self::Shape>>) -> Vec<GetSObj<Self::Shape>> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    self.sample_obj(thread_rng, info.clone())
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
                    self.sample_raw_obj(thread_rng)
                },
            )
            .collect()
    }

    /// Sample multiple optimizer-side solutions using per-thread RNGs.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_opt<R: Rng>(&self, rng: &mut R, size: usize, info: Arc<GetSInfo<Self::Shape>>) -> Vec<GetSOpt<Self::Shape>> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    self.sample_opt(thread_rng, info.clone())
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
                    self.sample_raw_opt(thread_rng)
                },
            )
            .collect()
    }

    /// Sample an objective-side solution and map it to the optimizer space.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<GetSInfo<Self::Shape>>) -> Self::Shape {
        self.onto_opt(self.sample_obj(rng, info.clone()))
    }

    /// Sample multiple pairs by sampling objective-side solutions then mapping them.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_pair<R: Rng>(&self,rng: &mut R,size: usize,info: Arc<GetSInfo<Self::Shape>>) -> Vec<Self::Shape> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                rand::make_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    self.sample_pair(thread_rng, info.clone())
                },
            )
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_obj<A, R>(&self,f: A,rng: &mut R,size: usize,info: Arc<GetSInfo<Self::Shape>>) -> Vec<GetSObj<Self::Shape>>
    where
        A: Fn(GetSObj<Self::Shape>) -> GetSObj<Self::Shape>,
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
                    f(self.sample_obj(thread_rng, info.clone()))
                },
            )
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_opt<A, R>(&self, f: A, rng: &mut R, size: usize, info: Arc<GetSInfo<Self::Shape>>) -> Vec<GetSOpt<Self::Shape>>
    where
        A: Fn(GetSOpt<Self::Shape>) -> GetSOpt<Self::Shape>,
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
                    f(self.sample_opt(thread_rng, info.clone()))
                },
            )
            .collect()
    }
}

/// Parallel [`Searchspace`] implementation for objective-only domains.
impl<F, Dom> Searchspace<F> for SpPar<Dom, NoDomain>
where
    F: FuncWrapper<Dom = Dom>,
    Dom: Domain,
    GetRaw<F::Sol<Dom>>: Uncomputed<X = Arc<[Dom::Element]>>,
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
        self.sample_raw_obj(rng)
    }

    /// Check whether a solution belongs to the objective domain.
    /// See module-level examples in [`crate::searchspace`].
    fn contains_obj(&self, inp: &GetRaw<F::Sol<Dom>>) -> bool {
        let variter = self.var.par_iter();
        inp.ref_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    /// Check whether a solution belongs to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn contains_opt(&self, inp: &GetRaw<F::Sol<Dom>>) -> bool {
        let variter = self.var.par_iter();
        inp.ref_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    /// Wrap a vector of objective-side solutions as lone solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_obj(&self, inp: Vec<F::Sol<Dom>>) -> Vec<Self::Shape> {
        inp.into_par_iter().map(|sol| Lone::new(sol)).collect()
    }

    /// Wrap a vector of objective-side solutions as lone solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_opt(&self, inp: Vec<F::Sol<Dom>>) -> Vec<Self::Shape> {
        inp.into_par_iter().map(|sol| Lone::new(sol)).collect()
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
                    self.sample_obj(thread_rng, info.clone())
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
                    self.sample_raw_obj(thread_rng)
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
                    self.sample_opt(thread_rng, info.clone())
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
                    self.sample_raw_opt(thread_rng)
                },
            )
            .collect()
    }

    /// Sample an objective-side solution and wrap it as a lone solution.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<GetSInfo<F::Sol<Dom>>>) -> Self::Shape {
        Lone::new(self.sample_opt(rng, info))
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
                    self.sample_pair(thread_rng, info.clone())
                },
            )
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_obj<A, R>(&self,f: A,rng: &mut R,size: usize,info: Arc<GetSInfo<F::Sol<Dom>>>) -> Vec<F::Sol<Dom>>
    where
        A: Fn(F::Sol<Dom>) -> F::Sol<Dom>,
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
                    f(self.sample_obj(thread_rng, info.clone()))
                },
            )
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_opt<A, R>(&self, f: A, rng: &mut R, size: usize, info: Arc<GetSInfo<F::Sol<Dom>>>) -> Vec<F::Sol<Dom>>
    where
        A: Fn(F::Sol<Dom>) -> F::Sol<Dom>,
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
                    f(self.sample_opt(thread_rng, info.clone()))
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

// /// CSV support for objective and optimizer sides of the parallel searchspace.
// impl<Obj, Opt> CSVLeftRight<SpPar<Obj, Opt>, Arc<[LinkTyObj<Self>]>, Arc<[LinkTyOpt<Self>]>>
//     for SpPar<Obj, Opt>
// where
//     Obj: OntoDom<Opt> + CSVWritable<(), LinkTyObj<Self>>,
//     Opt: OntoDom<Obj> + CSVWritable<(), LinkTyOpt<Self>>,
//     Var<Obj, Opt>: CSVLeftRight<Var<Obj, Opt>, LinkTyObj<Self>, LinkTyOpt<Self>>,
// {
//     /// Build the CSV header by concatenating variable headers.
//     /// See module-level examples in [`crate::searchspace`].
//     fn header(elem: &SpPar<Obj, Opt>) -> Vec<String> {
//         elem.var.iter().flat_map(Var::<Obj, Opt>::header).collect()
//     }

//     /// Write objective-side values as CSV columns.
//     /// See module-level examples in [`crate::searchspace`].
//     fn write_left(&self, comp: &Arc<[LinkTyObj<Self>]>) -> Vec<String> {
//         let var_iter = self.var.iter();
//         comp.iter()
//             .zip(var_iter)
//             .flat_map(|(c, v)| v.write_left(c))
//             .collect()
//     }

//     /// Write optimizer-side values as CSV columns.
//     /// See module-level examples in [`crate::searchspace`].
//     fn write_right(&self, comp: &Arc<[LinkTyOpt<Self>]>) -> Vec<String> {
//         let var_iter = self.var.iter();
//         comp.iter()
//             .zip(var_iter)
//             .flat_map(|(c, v)| v.write_right(c))
//             .collect()
//     }
// }

// /// CSV support for objective-only parallel searchspaces.
// impl<Obj> CSVLeftRight<SpPar<Obj, NoDomain>, Arc<[LinkTyObj<Self>]>, Arc<[LinkTyOpt<Self>]>>
//     for SpPar<Obj, NoDomain>
// where
//     Obj: Domain + CSVWritable<(), LinkTyObj<Self>>,
//     Var<Obj, NoDomain>: CSVLeftRight<Var<Obj, NoDomain>, LinkTyObj<Self>, LinkTyOpt<Self>>,
// {
//     /// Build the CSV header by concatenating variable headers.
//     /// See module-level examples in [`crate::searchspace`].
//     fn header(elem: &SpPar<Obj, NoDomain>) -> Vec<String> {
//         elem.var
//             .iter()
//             .flat_map(Var::<Obj, NoDomain>::header)
//             .collect()
//     }

//     /// Write objective-side values as CSV columns.
//     /// See module-level examples in [`crate::searchspace`].
//     fn write_left(&self, comp: &Arc<[LinkTyObj<Self>]>) -> Vec<String> {
//         let var_iter = self.var.iter();
//         comp.iter()
//             .zip(var_iter)
//             .flat_map(|(c, v)| v.write_left(c))
//             .collect()
//     }

//     /// Write optimizer-side values as CSV columns (same as objective-side in this mode).
//     /// See module-level examples in [`crate::searchspace`].
//     fn write_right(&self, comp: &Arc<[LinkTyOpt<Self>]>) -> Vec<String> {
//         let var_iter = self.var.iter();
//         comp.iter()
//             .zip(var_iter)
//             .flat_map(|(c, v)| v.write_right(c))
//             .collect()
//     }
// }
