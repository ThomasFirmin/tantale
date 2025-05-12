//! # Variable
//! This crate describes what a variable is. A [`Var`] is mostly used to link
//! two [`Domains`](crate::core::domain::Domain) together, the one
//! of the [`Objective`](crate::core::objective::Objective) [`Domain`](crate::core::domain::Domain) (`Obj`) function, and the one
//! of the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`](crate::core::domain::Domain) (`Opt`).
//! The  [`Var`] struct allows a certain flexibility of this link.
//! First, one can define custom [`sampler`](crate::core::domain::sampler) functions and link it to a [`Domains`](crate::core::domain::Domain).
//! Moreover, one can also define custom [`Onto`](crate::core::onto::Onto) functions to map `Opt` onto `Obj`, and conversely.
//! A [`Var`] is named via a [`static`] [`str`], that will be used within the save files.
//! 
//! A helper macro [`var!`](crate::core::variable::vmacros::var) with a custom syntax, can be used to help
//! creating a single variable. But be careful, it does not replace the procedural macro [`sp!`](../../../tantale/macros/macro.sp.html).
//! Indeed, [`sp!`](../../../tantale/macros/macro.sp.html) is able to handle [`Mixed`](crate::core::domain::Mixed) domains,
//! by automatically creating different `enum` structures,
//! and by wrapping [`sampler`](crate::core::domain::sampler) and [`Onto`](crate::core::onto::Onto) functions
//! into new functions.
//! 
//! # Example
//! 
//! ```
//! use tantale::core::{
//!     domain::{
//!         {Real, Unit, Domain},
//!         onto::Onto,
//!         sampler::{uniform_real, uniform_unit} // mostly used for examples},
//!        },
//!     variable::var::Var,
//! };
//! use std::rc::Rc;
//! 
//! let dom_obj = Rc::new(Real::new(0.0,100.0));
//! let dom_opt = Rc::new(Unit::new());
//! let v = Var{
//!     name : ("a", None),
//!     domain_obj : dom_obj,
//!     domain_opt : dom_opt,
//!     sampler_obj : uniform_real, 
//!     sampler_opt : uniform_unit,
//!     onto_obj_fn : Unit::onto, // Unit -> Real
//!     onto_opt_fn : Real::onto, // Real -> Unit
//! };
//! 
//! let mut rng = rand::rng();
//! let sample_obj = v.sample_obj(&mut rng);
//! let sample_opt = v.sample_opt(&mut rng);
//! let mapped_to_obj = v.onto_obj(&sample_opt);
//! let mapped_to_opt = v.onto_opt(&sample_obj);
//! 
//! println!(" OBJ : {} => OPT {}", sample_obj, mapped_to_opt.unwrap());
//! println!(" OPT : {} => OBJ {}", sample_opt, mapped_to_obj.unwrap());
//! 
//! // A var can be replicated using a Range (which changes the index in name.1)
//! 
//! let replicated = v.replicate(1..10);
//! 
//! for r in replicated{
//!     println!("({},{})",r.name.0, r.name.1.unwrap());
//! }
//! 
//! ```

#[doc(alias = "Variable")]
#[cfg(doc)]
use crate::objective::Objective;
#[cfg(doc)]
use crate::optimizer::Optimizer;
use crate::domain::{derrors::DomainError, onto::Onto, Domain};

use rand::prelude::ThreadRng;
use std::fmt::{Debug, Display};
use std::rc::Rc;

/// Describes a [`Var`] with an [`Objective`] [`Domain`]  and an [`Optimizer`] [`Domain`].
///
#[derive(Clone)]
pub struct Var<Obj, Opt = Obj>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    pub name: (&'static str, Option<usize>), // NAME + SUFFIX
    pub domain_obj: Rc<Obj>,
    pub domain_opt: Rc<Opt>,
    pub sampler_obj: fn(&Obj, &mut ThreadRng) -> Obj::TypeDom,
    pub sampler_opt: fn(&Opt, &mut ThreadRng) -> Opt::TypeDom,
    pub onto_obj_fn: fn(&Opt, &Opt::TypeDom, &Obj) -> Result<Obj::TypeDom, DomainError>,
    pub onto_opt_fn: fn(&Obj, &Obj::TypeDom, &Opt) -> Result<Opt::TypeDom, DomainError>,
}

/// Onto function when only the [`Objective`] [`Domain`] is define.
/// In that case, there is no need to map an input to the [`Optimizer`] [`Domain`].
///
pub fn _single_onto<T>(
    _input: &T,
    item: &T::TypeDom,
    _output: &T,
) -> Result<T::TypeDom, DomainError>
where
    T: Domain + Clone + Display + Debug,
{
    Ok(item.clone())
}

impl<'a, Obj> Var<Obj>
where
    Obj: Domain + Clone + Display + Debug,
{
    /// Creates a new instance of a [`Var`] when only the [`Objective`] [`Domain`] is defined.
    ///
    /// # Parameters
    ///
    /// * `name` : `&'a str` - Name of the Var.
    /// The name of the Var, mostly used for saving, or pass a point as a keyword.
    /// * `domobj` : [`Rc`]`<Obj>` - Accessible via the method [`domain_obj()`](Var::domain_obj).
    /// The [`Domain`] of the [`Objective`] [`Domain`].
    /// * `sampobj` : [`Option`]`<fn(&Obj, &mut `[`ThreadRng`]`) -> Obj::`[`TypeDom`](Domain::TypeDom)`>` -
    /// An optional sampler function for the [`Objective`] [`Domain`].
    /// By default uses the [`default_sampler`](Domain::default_sampler) of the [`Domain`].
    /// * sampopt : [`Option`]`<fn(&Opt, &mut `[`ThreadRng`]`) -> Opt::`[`TypeDom`](Domain::TypeDom)`>` -
    /// An optional sampler function for the [`Optimizer`] [`Domain`].
    /// By default uses the [`default_sampler`](Domain::default_sampler) of the [`Domain`].
    ///
    pub fn new_single(
        name: &'static str,
        domobj: Rc<Obj>,
        sampobj: Option<fn(&Obj, &mut ThreadRng) -> <Obj as Domain>::TypeDom>,
        sampopt: Option<fn(&Obj, &mut ThreadRng) -> <Obj as Domain>::TypeDom>,
    ) -> Var<Obj> {
        let samplerobj_selected = sampobj.unwrap_or(Obj::sample);
        let sampleropt_selected = sampopt.unwrap_or(Obj::sample);

        Var {
            name: (name, None),
            domain_obj: domobj.clone(),
            domain_opt: domobj,
            sampler_obj: samplerobj_selected,
            sampler_opt: sampleropt_selected,
            onto_obj_fn: _single_onto,
            onto_opt_fn: _single_onto,
        }
    }
}

impl<'a, Obj, Opt> Var<Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug + Onto<Opt>,
    Opt: Domain + Clone + Display + Debug + Onto<Obj>,
{
    /// Creates a new instance of a [`Var`] when the [`Objective`] and [`Optimizer`] [`Domain`]s are defined.
    ///
    /// # Parameters
    ///
    /// * `name` : `&'a str` - Name of the Var.
    /// The name of the Var, mostly used for saving, or pass a point as a keyword.
    /// * `domobj` : [`Rc`]`<Obj>` - Accessible via the method [`domain_obj()`](Var::domain_obj).
    /// The [`Domain`] of the [`Objective`] [`Domain`].
    /// * `domopt` : [`Rc`]`<Opt>` - Accessible via the method [`domain_opt()`](Var::domain_opt).
    /// The [`Domain`] of the [`Optimizer`] [`Domain`].
    /// * `sampobj` : [`Option`]`<fn(&Obj, &mut `[`ThreadRng`]`) -> Obj::`[`TypeDom`](Domain::TypeDom)`>` -
    /// An optional sampler function for the [`Objective`] [`Domain`].
    /// By default uses the [`default_sampler`](Domain::default_sampler) of the [`Domain`].
    /// * sampopt : [`Option`]`<fn(&Opt, &mut `[`ThreadRng`]`) -> Opt::`[`TypeDom`](Domain::TypeDom)`>` -
    /// An optional sampler function for the [`Optimizer`] [`Domain`].
    /// By default uses the [`default_sampler`](Domain::default_sampler) of the [`Domain`].
    ///
    pub fn new_double(
        name: &'static str,
        domobj: Rc<Obj>,
        domopt: Rc<Opt>,
        sampobj: Option<fn(&Obj, &mut ThreadRng) -> <Obj as Domain>::TypeDom>,
        sampopt: Option<fn(&Opt, &mut ThreadRng) -> <Opt as Domain>::TypeDom>,
    ) -> Var<Obj, Opt> {
        let samplerobj_selected = sampobj.unwrap_or(Obj::sample);
        let sampleropt_selected = sampopt.unwrap_or(Opt::sample);

        Var {
            name: (name, None),
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
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    pub fn onto_obj(&self, item: &Opt::TypeDom) -> Result<<Obj as Domain>::TypeDom, DomainError> {
        (self.onto_obj_fn)(&self.domain_opt, &item, &self.domain_obj)
    }
    pub fn onto_opt(&self, item: &Obj::TypeDom) -> Result<<Opt as Domain>::TypeDom, DomainError> {
        (self.onto_opt_fn)(&self.domain_obj, &item, &self.domain_opt)
    }
    pub fn sample_obj(&self, rng: &mut ThreadRng) -> <Obj as Domain>::TypeDom {
        (self.sampler_obj)(&self.domain_obj, rng)
    }
    pub fn sample_opt(&self, rng: &mut ThreadRng) -> <Opt as Domain>::TypeDom {
        (self.sampler_opt)(&self.domain_opt, rng)
    }
    pub fn replicate(&self, range : std::ops::Range<usize>) -> Vec<Self>{
        let mut vec = Vec::new();
        for i in range{

            let var = Self{
                name : (self.name.0, Some(i)),
                domain_obj : self.domain_obj.clone(),
                domain_opt : self.domain_opt.clone(),
                sampler_obj : self.sampler_obj,
                sampler_opt : self.sampler_opt,
                onto_obj_fn : self.onto_obj_fn,
                onto_opt_fn : self.onto_opt_fn,
            };
            vec.push(var);
        }
        vec

    }
}
