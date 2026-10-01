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
    FuncWrapper, HasId, HasInfo, HasVariables, HasX, aliases::{GetRaw, GetSInfo, GetSObj, GetSOpt}, domain::{
        Domain, NoDomain, PreDomain, onto::{Linked, OntoDom},
    }, searchspace::{Searchspace, SpPar}, solution::{
        Lone, Pair, Uncomputed,
    }, variable::{Var, var::ToName},
};

use rand::prelude::Rng;
use std::sync::Arc;

/// A basic [`Searchspace`] made of a boxed slice of [`Var`].
pub struct Sp<Obj: Domain, Opt: PreDomain> {
    /// Variables that define the [`Objective`](crate::Objective) and [`Optimizer`](crate::Optimizer) domains.
    pub var: Box<[Var<Obj, Opt>]>,
}

impl <Obj: Domain, Opt: PreDomain> Sp<Obj, Opt> {
    /// Create a new [`Sp`] from a boxed slice of [`Var`].
    pub fn new(var: Box<[Var<Obj, Opt>]>) -> Self {
        Self { var }
    }
    /// Get the number of variables in the searchspace.
    pub fn size(&self) -> usize {
        self.var.len()
    }
}

/// Link objective and optimizer domains when both sides are present.
impl<Obj: OntoDom<Opt>, Opt: OntoDom<Obj>> Linked for Sp<Obj, Opt> {
    type Obj = Obj;
    type Opt = Opt;
    type TrueOpt = Opt;
}

/// Link objective and optimizer domains (objective domains with itself) when using [`NoDomain`].
impl<Obj: Domain> Linked for Sp<Obj, NoDomain> {
    type Obj = Obj;
    type Opt = Obj;
    type TrueOpt = NoDomain;
}

impl<Obj, Opt> HasVariables for Sp<Obj, Opt>
where
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
{
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
    
    fn get_vars(&self) -> &[Var<Self::Obj, Self::Opt>] {
        &self.var
    }
}

impl<Obj: Domain> HasVariables for Sp<Obj, NoDomain> {
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
    
    fn get_vars(&self) -> &[Var<Obj, NoDomain>] {
        self.var.as_ref()
    }
}

/// [`Searchspace`] implementation for paired objective/optimizer domains.
impl<F, Obj, Opt> Searchspace<F> for Sp<Obj, Opt>
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
    fn onto_obj(&self, inp: GetSOpt<Self::Shape>) -> Self::Shape {
        let outx: Vec<Obj::Element> = self
            .var
            .iter()
            .zip(inp.ref_x().iter())
            .map(|(v, xi)| v.onto_obj(xi).unwrap())
            .collect();
        let solobj = GetSObj::<Self::Shape>::new(inp.id(), outx, inp.sinfo());
        Pair::new(solobj, inp)
    }

    /// Map an objective-side solution into the optimizer-side space. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn onto_opt(&self, inp: GetSObj<Self::Shape>) -> Self::Shape {
        let outx: Vec<Opt::Element> = self
            .var
            .iter()
            .zip(inp.ref_x().iter())
            .map(|(v, xi)| v.onto_opt(xi).unwrap())
            .collect();
        let solopt = GetSOpt::<Self::Shape>::new(inp.id(), outx, inp.sinfo());
        Pair::new(inp, solopt)
    }

    fn sample_raw_obj<R: Rng>(&self, rng: &mut R) -> GetRaw<GetSObj<Self::Shape>> {
        self.var.iter().map(|v| v.sample_obj(rng)).collect()
    }

    fn sample_raw_opt<R: Rng>(&self, rng: &mut R) -> GetRaw<GetSOpt<Self::Shape>> {
        self.var.iter().map(|v| v.sample_opt(rng)).collect()
    }

    /// Check whether a solution belongs to the objective domain.
    /// See module-level examples in [`crate::searchspace`].
    fn contains_obj(&self, inp: &GetRaw<GetSObj<Self::Shape>>) -> bool {
        inp.iter()
            .zip(&self.var)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    /// Check whether a solution belongs to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn contains_opt(&self, inp: &GetRaw<GetSOpt<Self::Shape>>) -> bool {
        inp.iter()
            .zip(&self.var)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    /// Map a vector of optimizer-side solutions into the objective space. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_obj(&self, inp: Vec<GetSOpt<Self::Shape>>) -> Vec<Self::Shape> {
        inp.into_iter().map(|i| <Sp<Obj, Opt> as Searchspace<F>>::onto_obj(self, i)).collect()
    }

    /// Map a vector of objective-side solutions into the optimizer space. See [`Onto`](crate::Onto) for details.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_opt(&self, inp: Vec<GetSObj<Self::Shape>>) -> Vec<Self::Shape> {
        inp.into_iter().map(|i| <Sp<Obj, Opt> as Searchspace<F>>::onto_opt(self, i)).collect()
    }

    /// Sample multiple objective-side solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_obj<R: Rng>(&self,rng: &mut R,size: usize, info: Arc<GetSInfo<Self::Shape>>) -> Vec<GetSObj<Self::Shape>> {
        (0..size)
            .map(|_| <Sp<Obj, Opt> as Searchspace<F>>::sample_obj::<R>(self, rng, info.clone()))
            .collect()
    }

    /// Sample multiple raw objective-side solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_raw_obj<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<GetRaw<GetSObj<Self::Shape>>> {
        (0..size)
            .map(|_| <Sp<Obj, Opt> as Searchspace<F>>::sample_raw_obj::<R>(self, rng))
            .collect()
    }

    /// Sample multiple optimizer-side solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_opt<R: Rng>(&self, rng: &mut R, size: usize, info: Arc<GetSInfo<Self::Shape>>) -> Vec<GetSOpt<Self::Shape>> {
        (0..size)
            .map(|_| <Sp<Obj, Opt> as Searchspace<F>>::sample_opt::<R>(self, rng, info.clone()))
            .collect()
    }

    fn vec_sample_raw_opt<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<GetRaw<GetSOpt<Self::Shape>>> {
        (0..size)
            .map(|_| <Sp<Obj, Opt> as Searchspace<F>>::sample_raw_opt::<R>(self, rng))
            .collect()
    }
    
    /// Sample an objective-side solution and map it to the optimizer space.
    /// See module-level examples in [`crate::searchspace`].
    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<GetSInfo<Self::Shape>>) -> Self::Shape {
        <Sp<Obj, Opt> as Searchspace<F>>::onto_opt(self, <Sp<Obj, Opt> as Searchspace<F>>::sample_obj::<R>(self, rng, info.clone()))
    }

    /// Sample multiple [`SolutionShape`](crate::solution::SolutionShape)
    /// by sampling objective-side solutions then mapping them to the optimizer space.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_pair<R: Rng>(&self,rng: &mut R,size: usize,info: Arc<GetSInfo<Self::Shape>>) -> Vec<Self::Shape> {
        (0..size)
            .map(|_| <Sp<Obj, Opt> as Searchspace<F>>::sample_pair(self, rng, info.clone()))
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_obj<A, R>(&self,f: A,rng: &mut R,size: usize,info: Arc<GetSInfo<Self::Shape>>) -> Vec<GetSObj<Self::Shape>>
    where
        A: Fn(GetSObj<Self::Shape>) -> GetSObj<Self::Shape>,
        R: Rng,
    {
        (0..size)
            .map(|_| f(<Sp<Obj, Opt> as Searchspace<F>>::sample_obj::<R>(self, rng, info.clone())))
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_opt<A, R>(&self, f: A, rng: &mut R, size: usize, info: Arc<GetSInfo<Self::Shape>>) -> Vec<GetSOpt<Self::Shape>>
    where
        A: Fn(GetSOpt<Self::Shape>) -> GetSOpt<Self::Shape>,
        R: Rng,
    {
        (0..size)
            .map(|_| f(<Sp<Obj, Opt> as Searchspace<F>>::sample_opt::<R>(self, rng, info.clone())))
            .collect()
    }
}

/// [`Searchspace`] implementation for objective-only domains.
impl<F, Dom> Searchspace<F> for Sp<Dom, NoDomain>
where
    F: FuncWrapper<Dom = Dom>,
    F::Sol<Dom>: Uncomputed<X = Arc<[Dom::Element]>>,
    Dom: Domain,
{
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
        self.var.iter().map(|v| v.sample_obj(rng)).collect()
    }

    fn sample_raw_opt<R: Rng>(&self, rng: &mut R) -> GetRaw<F::Sol<Dom>> {
        self.var.iter().map(|v| v.sample_opt(rng)).collect()
    }

    /// Check whether a solution belongs to the objective domain.
    /// See module-level examples in [`crate::searchspace`].
    fn contains_obj(&self, inp: &GetRaw<GetSObj<Self::Shape>>) -> bool {
        inp.iter()
            .zip(&self.var)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    /// Check whether a solution belongs to the optimizer domain.
    /// See module-level examples in [`crate::searchspace`].
    fn contains_opt(&self, inp: &GetRaw<GetSOpt<Self::Shape >>) -> bool {
        inp.iter()
            .zip(&self.var)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    /// Wrap a vector of optimizer-side solutions as lone solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_obj(&self, inp: Vec<F::Sol<Dom>>) -> Vec<Self::Shape> {
        inp.into_iter().map(Lone::new).collect()
    }

    /// Wrap a vector of objective-side solutions as lone solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_onto_opt(&self, inp: Vec<F::Sol<Dom>>) -> Vec<Self::Shape> {
        inp.into_iter().map(Lone::new).collect()
    }

    /// Sample multiple objective-side solutions.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_obj<R: Rng>(&self,rng: &mut R,size: usize,info: Arc<GetSInfo<F::Sol<Dom>>>) -> Vec<F::Sol<Dom>> {
        (0..size)
            .map(|_| <Sp<Dom, NoDomain> as Searchspace<F>>::sample_obj::<R>(self, rng, info.clone()))
            .collect()
    }

    fn vec_sample_raw_obj<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<GetRaw<F::Sol<Dom>>> {
        (0..size)
            .map(|_| <Sp<Dom, NoDomain> as Searchspace<F>>::sample_raw_obj::<R>(self, rng))
            .collect()
    }

    /// Sample multiple optimizer-side solutions (same as objective-side in this mode).
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_opt<R: Rng>(&self,rng: &mut R,size: usize,info: Arc<GetSInfo<F::Sol<Dom>>>) -> Vec<F::Sol<Dom>> {
        (0..size)
            .map(|_| <Sp<Dom, NoDomain> as Searchspace<F>>::sample_opt::<R>(self, rng, info.clone()))
            .collect()
    }
    
    fn vec_sample_raw_opt<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<GetRaw<F::Sol<Dom>>> {
        (0..size)
            .map(|_| <Sp<Dom, NoDomain> as Searchspace<F>>::sample_raw_opt::<R>(self, rng))
            .collect()
    }

    /// Sample an objective-side solution and wrap it as a [`Lone`] [`SolutionShape`](crate::solution::SolutionShape).
    /// See module-level examples in [`crate::searchspace`].
    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<GetSInfo<F::Sol<Dom>>>) -> Self::Shape {
        Lone::new(<Sp<Dom, NoDomain> as Searchspace<F>>::sample_obj::<R>(self, rng, info.clone()))
    }

    /// Sample multiple [`Lone`] [`SolutionShape`](crate::solution::SolutionShape).
    /// See module-level examples in [`crate::searchspace`].
    fn vec_sample_pair<R: Rng>(&self,rng: &mut R,size: usize,info: Arc<GetSInfo<F::Sol<Dom>>>) -> Vec<Self::Shape> {
        (0..size)
            .map(|_| Lone::new(<Sp<Dom, NoDomain> as Searchspace<F>>::sample_obj::<R>(self, rng, info.clone())))
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_obj<A, R>(&self,f: A,rng: &mut R,size: usize,info: Arc<GetSInfo<F::Sol<Dom>>>) -> Vec<F::Sol<Dom>>
    where
        A: Fn(F::Sol<Dom>) -> F::Sol<Dom>,
        R: Rng,
    {
        (0..size)
            .map(|_| f(<Sp<Dom, NoDomain> as Searchspace<F>>::sample_obj::<R>(self, rng, info.clone())))
            .collect()
    }

    /// Equivalent to `vec_sample_opt` but applies a function to each sampled solution.
    /// See module-level examples in [`crate::searchspace`].
    fn vec_apply_opt<A, R>(&self, f: A, rng: &mut R, size: usize, info: Arc<GetSInfo<F::Sol<Dom>>>) -> Vec<F::Sol<Dom>>
    where
        A: Fn(F::Sol<Dom>) -> F::Sol<Dom>,
        R: Rng,
    {
        (0..size)
            .map(|_| f(<Sp<Dom, NoDomain> as Searchspace<F>>::sample_opt::<R>(self, rng, info.clone())))
            .collect()
    }
}

/// Convert a multi-threaded (via [`rayon`]) searchspace into its sequential counterpart.
impl<Obj: Domain, Opt: Domain> From<SpPar<Obj, Opt>> for Sp<Obj, Opt> {
    fn from(value: SpPar<Obj, Opt>) -> Self {
        Sp { var: value.var }
    }
}