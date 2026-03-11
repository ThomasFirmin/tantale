use crate::{
    domain::{ConcreteDomain, Domain, NoDomain, TypeDom, onto::OntoDom}, errors::OntoError};

use rand::prelude::ThreadRng;
use rayon::iter::IntoParallelIterator;
use std::iter::{FlatMap, RepeatN, repeat_n};

type OntoFunc<A, B,TypeA,TypeB> = fn(&A, &TypeA, &B) -> Result<TypeB, OntoError>;

/// Describes a [`Var`] with an [`Objective`](crate::core::objective::Objective) [`Domain`]  and an [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`].
pub struct Var<Obj, Opt>
where
    Obj: ConcreteDomain,
    Opt: Domain,
{
    pub name: (&'static str, Option<usize>), // NAME + SUFFIX
    pub domain_obj: Obj,
    pub domain_opt: Opt,
    pub replicate: usize,
}

impl<Obj> Var<Obj,NoDomain>
where
    Obj: ConcreteDomain,
{
    /// Creates a new instance of a [`Var`] when only the [`Objective`](crate::core::objective::Objective) [`Domain`] is defined.
    ///
    /// # Parameters
    ///
    /// * `name` : `(&'static str, Option<usize>)` - Name of the Var.
    ///   The name of the Var, mostly used for saving, or pass a point as a keyword.
    /// * `domobj` : [`Arc`]`<Obj>` - Accessible via the method [`domain_obj()`](Var::domain_obj).
    ///   The [`Domain`] of the [`Objective`](crate::core::objective::Objective) [`Domain`].
    /// * `sampobj` : [`Option`]`<fn(&Obj, &mut `[`ThreadRng`]`) -> Obj::`[`TypeDom`](Domain::TypeDom)`>` -
    ///   An optional sampler function for the [`Objective`](crate::core::objective::Objective) [`Domain`].
    ///   By default uses the [`sampler`](Domain::sample) of the [`Domain`].
    /// * sampopt : [`Option`]`<fn(&Opt, &mut `[`ThreadRng`]`) -> Opt::`[`TypeDom`](Domain::TypeDom)`>` -
    ///   An optional sampler function for the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`].
    ///   By default uses the [`sampler`](Domain::sample) of the [`Domain`].
    ///
    pub fn new_single(
        name: (&'static str, Option<usize>),
        domain_obj: Obj,
        replicate:usize,
    ) -> Var<Obj,NoDomain> {
        Var {
            name,
            domain_obj,
            domain_opt: NoDomain,
            replicate,
        }
    }
}

impl<Obj, Opt> Var<Obj, Opt>
where
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
{
    /// Creates a new instance of a [`Var`] when the [`Objective`](crate::core::objective::Objective) and [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`]s are defined.
    ///
    /// # Parameters
    ///
    /// * `name` : `&'a `[`str`] - Name of the Var.
    ///   The name of the   Var, mostly used for saving, or pass a point as a keyword.
    /// * `domobj` : [`Arc`]`<Obj>` - Accessible via the method [`domain_obj()`](Var::domain_obj).
    ///   The [`Domain`] of the [`Objective`](crate::core::objective::Objective) [`Domain`].
    /// * `domopt` : [`Arc`]`<Opt>` - Accessible via the method [`domain_opt()`](Var::domain_opt).
    ///   The [`Domain`] of the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`].
    /// * `sampobj` : [`Option`]`<fn(&Obj, &mut `[`ThreadRng`]`) -> Obj::`[`TypeDom`](Domain::TypeDom)`>` -
    ///   An optional sampler function for the [`Objective`](crate::core::objective::Objective) [`Domain`].
    ///   By default uses the [`sampler`](Domain::sample) of the [`Domain`].
    /// * sampopt : [`Option`]`<fn(&Opt, &mut `[`ThreadRng`]`) -> Opt::`[`TypeDom`](Domain::TypeDom)`>` -
    ///   An optional sampler function for the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`].
    ///   By default uses the [`sampler`](Domain::sample) of the [`Domain`].
    ///
    pub fn new_double(
        name: (&'static str, Option<usize>),
        domain_obj: Obj,
        domain_opt: Opt,
        replicate:usize,
    ) -> Var<Obj, Opt> {
        Var {
            name,
            domain_obj,
            domain_opt,
            replicate,
        }
    }

    /// Function to map an `item` from the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`] onto the [`Objective`](crate::core::objective::Objective) [`Domain`].
    /// The function uses the given `onto_obj_fn` attribute. By default it uses the corresponding [`Onto`] function,
    /// If the input and output [`Domains`](Domain) are the same, the input `item` is copied to the output of the function.
    ///
    /// # Parameters
    ///
    /// * `item` : `&Opt::`[`TypeDom`](Domain::TypeDom) - A reference to point sampled within the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`] to be
    ///   mapped onto the [`Objective`](crate::core::objective::Objective) [`Domain`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Domain},
    ///     var,
    /// };
    ///
    /// let domobj = Real::new(0.0,100.0);
    /// let domopt = Unit::new();
    /// let v = var!("a" ; domobj ; domopt);
    ///
    /// let point_opt = 0.9;
    /// let mapped_to_obj = v.onto_obj(&point_opt);
    ///
    /// println!(" OPT : {} => OBJ {}", point_opt, mapped_to_obj.unwrap());
    ///
    /// ```
    ///
    pub fn onto_obj(&self, item: &TypeDom<Opt>) -> Result<TypeDom<Obj>, OntoError> {
        self.domain_opt.onto(item,&self.domain_obj)
    }
    /// Function to map an `item` from the [`Objective`](crate::core::objective::Objective) [`Domain`] onto the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`].
    /// The function uses the given `onto_opt_fn` attribute. By default it uses the corresponding [`Onto`] function.
    /// If the input and output [`Domains`](Domain) are the same, the input `item` is copied to the output of the function.
    ///
    /// # Parameters
    ///
    /// * `item` : `Obj::`[`TypeDom`](Domain::TypeDom) - A reference to point sampled within the [`Objective`](crate::core::objective::Objective) [`Domain`] to be
    ///   mapped onto the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Domain},
    ///     var,
    /// };
    ///
    /// let domobj = Real::new(0.0,100.0);
    /// let domopt = Unit::new();
    /// let v = var!("a" ; domobj ; domopt);
    ///
    /// let point_obj = 50.0;
    /// let mapped_to_opt = v.onto_opt(&point_obj);
    ///
    /// println!(" OBJ : {} => OPT {}", point_obj, mapped_to_opt.unwrap());
    ///
    /// ```
    pub fn onto_opt(&self, item: &TypeDom<Obj>) -> Result<TypeDom<Opt>, OntoError> {
        self.domain_obj.onto(item,&self.domain_opt)
    }
    /// Function to sample a point from the [`Objective`](crate::core::objective::Objective) [`Domain`].
    ///
    /// # Parameters
    ///
    /// * `rng` : `&mut `[`ThreadRng`](rand::prelude::ThreadRng) - A RNG thread.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Domain},
    ///     var,
    /// };
    ///
    /// let domobj = Real::new(0.0,100.0);
    /// let domopt = Unit::new();
    /// let v = var!("a" ; domobj ; domopt);
    ///
    /// let mut rng = rand::rng();
    /// let point_obj = v.sample_obj(&mut rng);
    /// let mapped_to_opt = v.onto_opt(&point_obj);
    ///
    /// println!(" OBJ : {} => OPT {}", point_obj, mapped_to_opt.unwrap());
    ///
    /// ```
    pub fn sample_obj(&self, rng: &mut ThreadRng) -> TypeDom<Obj> {
        self.domain_obj.sample(rng)
    }

}

impl<Obj: ConcreteDomain, Opt:ConcreteDomain> Var<Obj, Opt>
{
    /// Function to sample a point from the [`Objective`](crate::core::objective::Objective) [`Domain`].
    ///
    /// # Parameters
    ///
    /// * `rng` : `&mut `[`ThreadRng`](rand::prelude::ThreadRng) - A RNG thread.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Domain},
    ///     var,
    /// };
    ///
    /// let domobj = Real::new(0.0,100.0);
    /// let domopt = Unit::new();
    /// let v = var!("a" ; domobj ; domopt);
    ///
    /// let mut rng = rand::rng();
    /// let point_obj = v.sample_obj(&mut rng);
    /// let mapped_to_opt = v.onto_opt(&point_obj);
    ///
    /// println!(" OBJ : {} => OPT {}", point_obj, mapped_to_opt.unwrap());
    ///
    /// ```
    pub fn sample_obj(&self, rng: &mut ThreadRng) -> Obj::TypeDom {
        self.domain_obj.sample(rng)
    }
    /// Function to sample a point from the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`].
    ///
    /// # Parameters
    ///
    /// * `rng` : `&mut `[`ThreadRng`](rand::prelude::ThreadRng) - A RNG thread.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Domain},
    ///     var,
    /// };
    ///
    /// let domobj = Real::new(0.0,100.0);
    /// let domopt = Unit::new();
    /// let v = var!("a" ; domobj ; domopt);
    ///
    /// let mut rng = rand::rng();
    /// let point_opt = v.sample_opt(&mut rng);
    /// let mapped_to_obj = v.onto_obj(&point_opt);
    ///
    /// println!(" OPT : {} => OBJ {}", point_opt, mapped_to_obj.unwrap());
    ///
    /// ```
    ///
    pub fn sample_opt(&self, rng: &mut ThreadRng) -> Opt::TypeDom {
        self.domain_opt.sample(rng)
    }
}

impl <'a,Obj:ConcreteDomain,Opt:Domain> IntoIterator for &'a Var<Obj,Opt>{
    type Item = &'a Var<Obj,Opt>;
    type IntoIter = RepeatN<&'a Var<Obj,Opt>>;

    fn into_iter(self) -> Self::IntoIter {
        repeat_n(self, self.replicate)
    }
}

impl <'a,Obj:ConcreteDomain + Send + Sync,Opt:Domain + Send + Sync> IntoParallelIterator for &'a Var<Obj,Opt>{
    type Iter=rayon::iter::RepeatN<&'a Var<Obj,Opt>>;
    type Item=&'a Var<Obj,Opt>;

    fn into_par_iter(self) -> Self::Iter {
        rayon::iter::repeat_n(self, self.replicate)
    }
}



/// A run-length encoded vector: stores `(value, count)` pairs,
/// but behaves like a flattened vector of repeated elements.
#[derive(Clone, Debug)]
struct CountSlice<'a, Obj:ConcreteDomain,Opt:Domain>{
    data: &'a [Var<Obj,Opt>],
    // optional: prefix sums for O(log N) indexing
    counts: Vec<usize>,
}

impl<'a, Obj:ConcreteDomain,Opt:Domain> CountSlice<'a, Obj,Opt>{
    /// Construct from a vector of `(value, count)` pairs
    fn new(data: Vec<Var<Obj,Opt>>) -> Self {
        let mut counts = Vec::with_capacity(data.len());
        let mut sum = 0;
        for &(_, count) in &data {
            sum += count;
            counts.push(sum);
        }
        Self { data, counts }
    }

    /// Total virtual length
    fn len(&self) -> usize {
        self.prefix_counts.last().copied().unwrap_or(0)
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// O(log N) get by virtual index
    fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            return None;
        }
        // Binary search in prefix_counts
        let pos = match self.prefix_counts.binary_search(&(index + 1)) {
            Ok(p) => p,
            Err(p) => p,
        };
        Some(&self.data[pos].0)
    }

    /// Iterator over virtual elements
    fn iter(&self) -> RLEVecIter<'_, T> {
        RLEVecIter {
            vec: self,
            outer_idx: 0,
            inner_idx: 0,
        }
    }
}

/// Index trait for convenient `rle[i]` syntax
impl<T> Index<usize> for RLEVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
            .expect("index out of bounds in RLEVec")
    }
}

pub trait Unroll
{
    type Item;
    type Iter: Iterator<Item = Self::Item>;
    fn unroll(&self) -> Self::Iter;
}

impl<'a, Obj:ConcreteDomain,Opt:Domain> Unroll for &'a [Var<Obj,Opt>]{
    type Item = &'a Var<Obj,Opt>;
    type Iter = FlatMap<std::slice::Iter<'a, Var<Obj, Opt>>, RepeatN<&'a Var<Obj, Opt>>, fn(&'a Var<Obj, Opt>) -> RepeatN<&'a Var<Obj, Opt>>>;

    fn unroll(&self) -> Self::Iter {
        self.iter().flat_map(|v| v.into_iter())
    }
}

impl<'a, Obj:ConcreteDomain,Opt:Domain> Unroll for &'a Vec<Var<Obj,Opt>>{
    type Item = &'a Var<Obj,Opt>;
    type Iter = FlatMap<std::slice::Iter<'a, Var<Obj, Opt>>, RepeatN<&'a Var<Obj, Opt>>, fn(&'a Var<Obj, Opt>) -> RepeatN<&'a Var<Obj, Opt>>>;

    fn unroll(&self) -> Self::Iter {
        self.iter().flat_map(|v| v.into_iter())
    }
}