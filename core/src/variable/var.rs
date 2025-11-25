use crate::{
    domain::{onto::Onto, Domain, TypeDom},
    recorder::csv::{CSVLeftRight, CSVWritable},
    errors::OntoError,
};

use rand::prelude::ThreadRng;
use std::sync::Arc;

type OntoFunc<A, B> = fn(&A, &TypeDom<A>, &B) -> Result<TypeDom<B>, OntoError>;

/// Describes a [`Var`] with an [`Objective`](crate::core::objective::Objective) [`Domain`]  and an [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`].
#[derive(Clone)]
pub struct Var<Obj, Opt = Obj>
where
    Obj: Domain,
    Opt: Domain,
{
    name: (&'static str, Option<usize>), // NAME + SUFFIX
    domain_obj: Arc<Obj>,
    domain_opt: Arc<Opt>,
    sampler_obj: fn(&Obj, &mut ThreadRng) -> TypeDom<Obj>,
    sampler_opt: fn(&Opt, &mut ThreadRng) -> TypeDom<Opt>,
    onto_obj_fn: OntoFunc<Opt, Obj>,
    onto_opt_fn: OntoFunc<Obj, Opt>,
}

/// Onto function when only the [`Objective`](crate::core::objective::Objective) [`Domain`] is define.
/// In that case, there is no need to map an input to the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`].
///
pub fn _single_onto<T>(_input: &T, item: &TypeDom<T>, _output: &T) -> Result<TypeDom<T>, OntoError>
where
    T: Domain,
{
    Ok(item.clone())
}

impl<Obj> Var<Obj>
where
    Obj: Domain,
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
        domobj: Arc<Obj>,
        sampobj: Option<fn(&Obj, &mut ThreadRng) -> TypeDom<Obj>>,
        sampopt: Option<fn(&Obj, &mut ThreadRng) -> TypeDom<Obj>>,
    ) -> Var<Obj> {
        let samplerobj_selected = sampobj.unwrap_or(Obj::sample);
        let sampleropt_selected = sampopt.unwrap_or(Obj::sample);

        Var {
            name,
            domain_obj: domobj.clone(),
            domain_opt: domobj,
            sampler_obj: samplerobj_selected,
            sampler_opt: sampleropt_selected,
            onto_obj_fn: _single_onto,
            onto_opt_fn: _single_onto,
        }
    }
}

impl<Obj, Opt> Var<Obj, Opt>
where
    Obj: Domain + Onto<Opt, TargetItem = TypeDom<Opt>, Item = TypeDom<Obj>>,
    Opt: Domain + Onto<Obj, TargetItem = TypeDom<Obj>, Item = TypeDom<Opt>>,
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
        domobj: Arc<Obj>,
        domopt: Arc<Opt>,
        sampobj: Option<fn(&Obj, &mut ThreadRng) -> TypeDom<Obj>>,
        sampopt: Option<fn(&Opt, &mut ThreadRng) -> TypeDom<Opt>>,
    ) -> Var<Obj, Opt> {
        let samplerobj_selected = sampobj.unwrap_or(Obj::sample);
        let sampleropt_selected = sampopt.unwrap_or(Opt::sample);

        Var {
            name,
            domain_obj: domobj,
            domain_opt: domopt,
            sampler_obj: samplerobj_selected,
            sampler_opt: sampleropt_selected,
            onto_obj_fn: Opt::onto,
            onto_opt_fn: Obj::onto,
        }
    }
}

impl<Obj, Opt> Var<Obj, Opt>
where
    Obj: Domain,
    Opt: Domain,
{
    pub fn _new(
        name: (&'static str, Option<usize>),
        domobj: Arc<Obj>,
        domopt: Arc<Opt>,
        sampobj: fn(&Obj, &mut ThreadRng) -> TypeDom<Obj>,
        sampopt: fn(&Opt, &mut ThreadRng) -> TypeDom<Opt>,
        ontoobj: OntoFunc<Opt, Obj>,
        ontoopt: OntoFunc<Obj, Opt>,
    ) -> Var<Obj, Opt> {
        Var {
            name,
            domain_obj: domobj,
            domain_opt: domopt,
            sampler_obj: sampobj,
            sampler_opt: sampopt,
            onto_obj_fn: ontoobj,
            onto_opt_fn: ontoopt,
        }
    }

    /// Getter method for the `name` attribute of the [`Var`].
    pub fn get_name(&self) -> (&'static str, Option<usize>) {
        self.name
    }
    /// Getter method for the `domain_obj` ([`Objective`](crate::core::objective::Objective) [`Domain`]) attribute of the [`Var`].
    pub fn get_domain_obj(&self) -> Arc<Obj> {
        self.domain_obj.clone()
    }
    /// Getter method for the `domain_opt` ([`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`]) attribute of the [`Var`].
    pub fn get_domain_opt(&self) -> Arc<Opt> {
        self.domain_opt.clone()
    }
    /// Getter method for the `sampler_obj` attribute of the [`Var`].
    pub fn get_sampler_obj(&self) -> fn(&Obj, &mut ThreadRng) -> TypeDom<Obj> {
        self.sampler_obj
    }
    /// Getter method for the `sampler_opt` attribute of the [`Var`].
    pub fn get_sampler_opt(&self) -> fn(&Opt, &mut ThreadRng) -> TypeDom<Opt> {
        self.sampler_opt
    }
    /// Getter method for the `onto_obj_fn` attribute of the [`Var`].This the function used to map
    /// a point from the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`] onto the [`Objective`](crate::core::objective::Objective) [`Domain`].
    pub fn get_onto_obj_fn(&self) -> OntoFunc<Opt, Obj> {
        self.onto_obj_fn
    }
    /// Getter method for the `onto_opt_fn` attribute of the [`Var`].This the function used to map
    /// a point from the [`Objective`](crate::core::objective::Objective) [`Domain`] onto the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`].
    pub fn get_onto_opt_fn(&self) -> OntoFunc<Obj, Opt> {
        self.onto_opt_fn
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
        (self.onto_obj_fn)(&self.domain_opt, item, &self.domain_obj)
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
        (self.onto_opt_fn)(&self.domain_obj, item, &self.domain_opt)
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
        (self.sampler_obj)(&self.domain_obj, rng)
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
    pub fn sample_opt(&self, rng: &mut ThreadRng) -> TypeDom<Opt> {
        (self.sampler_opt)(&self.domain_opt, rng)
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
        let mut vec = Vec::new();
        for i in 0..repeats {
            let var = Self::_new(
                (self.name.0, Some(i)),
                self.domain_obj.clone(),
                self.domain_opt.clone(),
                self.sampler_obj,
                self.sampler_opt,
                self.onto_obj_fn,
                self.onto_opt_fn,
            );
            vec.push(var);
        }
        vec
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
    pub fn is_in_obj(&self, item: &TypeDom<Obj>) -> bool {
        self.domain_obj.is_in(item)
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
    pub fn is_in_opt(&self, item: &TypeDom<Opt>) -> bool {
        self.domain_opt.is_in(item)
    }
}

impl<Obj, Opt> CSVLeftRight<Self, Obj::TypeDom, Opt::TypeDom> for Var<Obj, Opt>
where
    Obj: Domain + CSVWritable<(), Obj::TypeDom>,
    Opt: Domain + CSVWritable<(), Opt::TypeDom>,
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
