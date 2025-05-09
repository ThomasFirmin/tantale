#[cfg(doc)]
use crate::objective::Objective;
#[cfg(doc)]
use crate::optimizer::Optimizer;

/// # Variable
/// This crate describes what a variable is.
/// Most of the domains implements the [`Domain`] type trait [TypeDom](Domain::TypeDom).
/// It gives the type of a point within this domain.
/// [`Domains`] are use in [`Var`] to define the type of the Var for the input type of
/// that Var within the [`Objective`] function, and the input type of the [`Optimizer`].
///
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
    pub _onto_obj_fn: fn(&Opt, &Opt::TypeDom, &Obj) -> Result<Obj::TypeDom, DomainError>,
    pub _onto_opt_fn: fn(&Obj, &Obj::TypeDom, &Opt) -> Result<Opt::TypeDom, DomainError>,
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
            _onto_obj_fn: _single_onto,
            _onto_opt_fn: _single_onto,
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
            _onto_obj_fn: Opt::onto,
            _onto_opt_fn: Obj::onto,
        }
    }
}

impl<Obj, Opt> Var<Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    pub fn onto_obj(&self, item: &Opt::TypeDom) -> Result<<Obj as Domain>::TypeDom, DomainError> {
        (self._onto_obj_fn)(&self.domain_opt, &item, &self.domain_obj)
    }
    pub fn onto_opt(&self, item: &Obj::TypeDom) -> Result<<Opt as Domain>::TypeDom, DomainError> {
        (self._onto_opt_fn)(&self.domain_obj, &item, &self.domain_opt)
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
            let domain_obj = self.domain_obj.clone();
            let domain_opt = self.domain_opt.clone();
            let sampler_obj = self.sampler_obj;
            let sampler_opt = self.sampler_opt;
            let _onto_obj_fn = self._onto_obj_fn;
            let _onto_opt_fn = self._onto_opt_fn;
            
            let var = Self{
                name : (self.name.0, Some(i)),
                domain_obj,
                domain_opt,
                sampler_obj,
                sampler_opt,
                _onto_obj_fn,
                _onto_opt_fn,
            };
            vec.push(var);
        }
        vec

    }
}
