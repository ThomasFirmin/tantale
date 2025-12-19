use crate::{
    domain::{Domain, PreDomain, NoDomain, onto::{OntoDom, Linked, LinkObj, LinkOpt, LinkTyObj, LinkTyOpt}}, errors::OntoError, recorder::csv::{CSVLeftRight, CSVWritable}
};

use rand::prelude::ThreadRng;
use std::sync::Arc;

/// Describes a [`Var`] with an [`Objective`](crate::core::objective::Objective) [`Domain`]  and an [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`].
#[derive(Clone)]
pub struct Var<Obj, Opt>
where
    Obj: Domain,
    Opt: PreDomain,
{
    pub name: (&'static str, Option<usize>), // NAME + SUFFIX
    pub domain_obj: Arc<Obj>,
    pub domain_opt: Arc<Opt>,
}

impl<Obj:OntoDom<Opt>, Opt:OntoDom<Obj>> Linked for Var<Obj,Opt>
{
    type Obj = Obj;
    type Opt = Opt;
}

impl<Obj:Domain> Linked for Var<Obj,NoDomain>
{
    type Obj = Obj;
    type Opt = Obj;
}

impl<Obj:OntoDom<Opt>, Opt:OntoDom<Obj>> Var<Obj, Opt>
{
    /// Function to sample a point from the [`Objective`](crate::core::objective::Objective) [`Domain`].
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
    pub fn sample_obj(&self, rng: &mut ThreadRng) -> LinkTyObj<Self> {
        LinkObj::<Self>::sample(&self.domain_obj, rng)
    }
    /// Function to sample a point from the [`Objective`](crate::core::objective::Objective) [`Domain`].
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
    pub fn sample_opt(&self, rng: &mut ThreadRng) -> LinkTyOpt<Self> {
        LinkOpt::<Self>::sample(&self.domain_opt, rng)
    }
    /// Check if an `item` is in the `Obj` [`Domain`] of the [`Var`].
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
    /// let v = tantale::core::var!("a" ; domobj ; domopt);
    ///
    /// let mut rng = rand::rng();
    /// let point_obj = v.sample_obj(&mut rng);
    /// let mapped_to_opt = v.onto_opt(&point_obj);
    ///
    /// println!(" OBJ : {} => OPT {}", point_obj, mapped_to_opt.unwrap());
    ///
    /// ```
    pub fn is_in_obj(&self, item: &LinkTyObj<Self>) -> bool {
        LinkObj::<Self>::is_in(&self.domain_obj,item)
    }
    /// Check if an `item` is in the `Opt` [`Domain`] of the [`Var`].
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
    /// let v = tantale::core::var!("a" ; domobj ; domopt);
    ///
    /// let mut rng = rand::rng();
    /// let point_obj = v.sample_obj(&mut rng);
    /// let mapped_to_opt = v.onto_opt(&point_obj);
    ///
    /// println!(" OBJ : {} => OPT {}", point_obj, mapped_to_opt.unwrap());
    ///
    /// ```
    pub fn is_in_opt(&self, item: &LinkTyOpt<Self>) -> bool {
        LinkOpt::<Self>::is_in(&self.domain_opt,item)
    }
    /// Function to replicate a variable a certain number of times .
    /// A new [`Var`] struct is created by cloning the [`Arc`] references of the domain, and by incrementing the
    /// second part of the `name` tuple.
    ///
    /// # Notes
    ///
    /// Consumes Self.
    ///
    /// # Parameters
    ///
    /// * `repeats` : `usize` - Number of repetitions of the [`Var`].
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
    /// let vec_v = v.replicate(10);
    ///
    /// for var in vec_v{
    ///     let point_opt = var.sample_opt(&mut rng);
    ///     let mapped_to_obj = var.onto_obj(&point_opt);
    ///     println!(" OPT : {} => OBJ {}", point_opt, mapped_to_obj.unwrap());     
    /// }
    ///
    /// ```
    ///
    pub fn replicate(self, repeats: usize) -> Vec<Self> {
        let mut vec = Vec::with_capacity(repeats);
        for i in 0..repeats {
            let var = Var { 
                name: (self.name.0, Some(i)), 
                domain_obj: self.domain_obj.clone(), 
                domain_opt: self.domain_opt.clone() 
            };
            vec.push(var);
        }
        vec
    }
}

impl<Obj:Domain> Var<Obj, NoDomain>
{
    pub fn new(
        name: (&'static str, Option<usize>),
        domain_obj: Obj,
        domain_opt: NoDomain,
    ) -> Var<Obj, NoDomain> {
        Var {
            name,
            domain_obj: Arc::new(domain_obj),
            domain_opt: Arc::new(domain_opt),
        }
    }
     pub fn sample_obj(&self, rng: &mut ThreadRng) -> LinkTyObj<Self> {
        LinkObj::<Self>::sample(&self.domain_obj, rng)
    }
    pub fn sample_opt(&self, rng: &mut ThreadRng) -> LinkTyOpt<Self> {
        LinkOpt::<Self>::sample(&self.domain_obj, rng)
    }
    pub fn is_in_obj(&self, item: &LinkTyObj<Self>) -> bool {
        LinkObj::<Self>::is_in(&self.domain_obj,item)
    }
    pub fn is_in_opt(&self, item: &LinkTyOpt<Self>) -> bool {
        LinkOpt::<Self>::is_in(&self.domain_obj,item)
    }
    pub fn replicate(self, repeats: usize) -> Vec<Self> {
        let mut vec = Vec::with_capacity(repeats);
        for i in 0..repeats {
            let var = Var { 
                name: (self.name.0, Some(i)), 
                domain_obj: self.domain_obj.clone(), 
                domain_opt: self.domain_opt.clone() 
            };
            vec.push(var);
        }
        vec
    }
}

impl<Obj:Domain> Var<Obj, Obj>
{
    pub fn new(
        name: (&'static str, Option<usize>),
        domain_obj: Obj,
        domain_opt: NoDomain,
    ) -> Var<Obj, NoDomain> {
        Var {
            name,
            domain_obj: Arc::new(domain_obj),
            domain_opt: Arc::new(domain_opt),
        }
    }
     pub fn sample_obj(&self, rng: &mut ThreadRng) -> LinkTyObj<Self> {
        LinkObj::<Self>::sample(&self.domain_obj, rng)
    }
    pub fn sample_opt(&self, rng: &mut ThreadRng) -> LinkTyOpt<Self> {
        LinkOpt::<Self>::sample(&self.domain_obj, rng)
    }
    pub fn is_in_obj(&self, item: &LinkTyObj<Self>) -> bool {
        LinkObj::<Self>::is_in(&self.domain_obj,item)
    }
    pub fn is_in_opt(&self, item: &LinkTyOpt<Self>) -> bool {
        LinkOpt::<Self>::is_in(&self.domain_obj,item)
    }
    pub fn replicate(self, repeats: usize) -> Vec<Self> {
        let mut vec = Vec::with_capacity(repeats);
        for i in 0..repeats {
            let var = Var { 
                name: (self.name.0, Some(i)), 
                domain_obj: self.domain_obj.clone(), 
                domain_opt: self.domain_opt.clone() 
            };
            vec.push(var);
        }
        vec
    }
}

impl<Obj:OntoDom<Opt>, Opt:OntoDom<Obj>> Var<Obj, Opt>
{
    /// Creates a new instance of a [`Var`] when the [`Objective`](crate::core::objective::Objective)
    /// and [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`]s are defined.
    pub fn new(
        name: (&'static str, Option<usize>),
        domain_obj: Obj,
        domain_opt: Opt,
    ) -> Var<Obj, Opt> {
        Var {
            name,
            domain_obj: Arc::new(domain_obj),
            domain_opt: Arc::new(domain_opt),
        }
    }
    /// Function to map an `item` from the [`Optimizer`](crate::core:optimizer::Optimizer) [`Domain`]
    /// onto the [`Objective`](crate::core::objective::Objective) [`Domain`].
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
    pub fn onto_obj(&self, item: &LinkTyOpt<Self>) -> Result<LinkTyObj<Self>, OntoError>
    {
        self.domain_opt.onto(item, &self.domain_obj)
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
    pub fn onto_opt(&self, item: &LinkTyObj<Self>) -> Result<LinkTyOpt<Self>, OntoError> {
        self.domain_obj.onto(item, &self.domain_opt)
    }
}

impl<Obj, Opt> CSVLeftRight<Self, LinkTyObj<Self>, LinkTyOpt<Self>> for Var<Obj, Opt>
where
    Obj: OntoDom<Opt> + CSVWritable<(), LinkTyObj<Self>>,
    Opt: OntoDom<Obj> + CSVWritable<(), LinkTyOpt<Self>>,
{
    fn header(elem: &Self) -> Vec<String> {
        let (name, id) = elem.name;
        let name_str = match id {
            Some(i) => format!("{}{}", name, i),
            None => String::from(name),
        };
        let dom_spec = Obj::header(&());
        if dom_spec.is_empty() {
            vec![name_str]
        } else {
            dom_spec
                .iter()
                .map(|head| format!("{}{}", name_str, head))
                .collect()
        }
    }

    fn write_left(&self, comp: &Obj::TypeDom) -> Vec<String> {
        self.domain_obj.write(comp)
    }

    fn write_right(&self, comp: &Opt::TypeDom) -> Vec<String> {
        self.domain_opt.write(comp)
    }
}

impl<Obj> CSVLeftRight<Self, LinkTyObj<Self>, LinkTyOpt<Self>> for Var<Obj,NoDomain>
where
    Obj: Domain + CSVWritable<(), LinkTyObj<Self>>,
{
    fn header(elem: &Self) -> Vec<String> {
        let (name, id) = elem.name;
        let name_str = match id {
            Some(i) => format!("{}{}", name, i),
            None => String::from(name),
        };
        let dom_spec = Obj::header(&());
        if dom_spec.is_empty() {
            vec![name_str]
        } else {
            dom_spec
                .iter()
                .map(|head| format!("{}{}", name_str, head))
                .collect()
        }
    }

    fn write_left(&self, comp: &LinkTyObj<Self>) -> Vec<String> {
        self.domain_obj.write(comp)
    }

    fn write_right(&self, comp: &LinkTyOpt<Self>) -> Vec<String> {
        self.domain_obj.write(comp)
    }
}
