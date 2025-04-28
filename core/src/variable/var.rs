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
use crate::domain::{
    onto::Onto,
    Domain,
    derrors::DomainError
};

use rand::prelude::ThreadRng;
use std::fmt::{Debug, Display};
use std::rc::Rc;

/// Describes a [`Var`] with an [`Objective`] [`Domain`]  and an [`Optimizer`] [`Domain`].
///
#[derive(Clone)]
pub struct Var<'a, Obj, Opt = Obj>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    pub name: &'a str,
    pub domain_obj: Rc<Obj>,
    pub domain_opt: Rc<Opt>,
    pub sampler_obj: fn(&Obj, &mut ThreadRng) -> Obj::TypeDom,
    pub sampler_opt: fn(&Opt, &mut ThreadRng) -> Opt::TypeDom,
    _onto_obj_fn: fn(&Opt, &Opt::TypeDom, &Obj) -> Result<Obj::TypeDom, DomainError>,
    _onto_opt_fn: fn(&Obj, &Obj::TypeDom, &Opt) -> Result<Opt::TypeDom, DomainError>,
}

/// Onto function when only the [`Objective`] [`Domain`] is define.
/// In that case, there is no need to map an input to the [`Optimizer`] [`Domain`].
///
fn _single_onto<T>(_input: &T, item: &T::TypeDom, _output: &T) -> Result<T::TypeDom, DomainError>
where
    T: Domain + Clone + Display + Debug,
{
    Ok(item.clone())
}

impl<'a, Obj> Var<'a, Obj>
where
    Obj: Domain + Clone + Display + Debug,
{
    //// Creates a new instance of a [`Var`] when only the [`Objective`] [`Domain`] is defined.
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
        name: &'a str,
        domobj: Rc<Obj>,
        sampobj: Option<fn(&Obj, &mut ThreadRng) -> <Obj as Domain>::TypeDom>,
        sampopt: Option<fn(&Obj, &mut ThreadRng) -> <Obj as Domain>::TypeDom>,
    ) -> Var<'a, Obj> {
        let samplerobj_selected = sampobj.unwrap_or(Obj::sample);
        let sampleropt_selected = sampopt.unwrap_or(Obj::sample);

        Var {
            name: name,
            domain_obj: domobj.clone(),
            domain_opt: domobj,
            sampler_obj: samplerobj_selected,
            sampler_opt: sampleropt_selected,
            _onto_obj_fn: _single_onto,
            _onto_opt_fn: _single_onto,
        }
    }
}

impl<'a, Obj, Opt> Var<'a, Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug + Onto<Opt>,
    Opt: Domain + Clone + Display + Debug + Onto<Obj>,
{
    //// Creates a new instance of a [`Var`] when the [`Objective`] and [`Optimizer`] [`Domain`]s are defined.
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
        name: &'a str,
        domobj: Rc<Obj>,
        domopt: Rc<Opt>,
        sampobj: Option<fn(&Obj, &mut ThreadRng) -> <Obj as Domain>::TypeDom>,
        sampopt: Option<fn(&Opt, &mut ThreadRng) -> <Opt as Domain>::TypeDom>,
    ) -> Var<'a, Obj, Opt> {
        let samplerobj_selected = sampobj.unwrap_or(Obj::sample);
        let sampleropt_selected = sampopt.unwrap_or(Opt::sample);

        Var {
            name: name,
            domain_obj: domobj,
            domain_opt: domopt,
            sampler_obj: samplerobj_selected,
            sampler_opt: sampleropt_selected,
            _onto_obj_fn: |obj, item, opt| Opt::onto(obj, item, opt),
            _onto_opt_fn: Obj::onto,
        }
    }
}

impl<'a, Obj, Opt> Var<'a, Obj, Opt>
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
}

// pub trait IntoBase<'a, Obj, Opt=Obj>
// where
//     Obj : Domain + Clone + Display + Debug + Into<BaseDom>,
//     Opt : Domain + Clone + Display + Debug + Into<BaseDom>,
// {
//     fn into_obj_base(self)->Var<'a, BaseDom, Opt>;
//     fn into_opt_base(self)->Var<'a, Obj, BaseDom>;
//     fn into_single_base(self)->Var<'a, BaseDom>;
// }

// impl IntoBase for Var<'a, Real>
// {
//     fn into_single_base(self, wrapped:)->Var<'a, BaseDom> {
//         let domobj = *self.domain_obj;
//         let domobj:BaseDom = domobj.into();
//         let domobj = Rc::new(domobj);

//         Var{
//                     name:self.name,
//                     domain_obj:domobj.clone(),
//                     domain_opt:domobj,
//                     sampler_obj: self.sampler_obj,
//                     sampler_opt: self.sampler_opt,
//                     onto_obj_fn: Opt::onto,
//                     onto_opt_fn:Obj::onto,
//                     _single_dom:true,
//                 }
//     }
// }

// impl<'a,Opt> Var<'a, Real, Opt>
// where
//     Opt: Domain + Clone + Display + Debug,
// {
//     pub fn into_obj_base(self)->Var<'a, BaseDom, Opt>{
//         let domobj = *self.domain_obj;
//         let domobj:BaseDom = domobj.into();
//         let sampler_obj = self.sampler_obj;
//         let sampler_obj = |domain,rng|{
//             match domain{
//                 BaseDom::Real(d) => BaseTypeDom::Real(sampler_obj(&d,rng)),
//                 _ => unreachable!("Can only wrap real sampler with wrap_real_sampler."),
//             }
//         };

//         if self._single_dom{
//             Var{
//                 name:self.name,
//                 domain_obj:domobj.clone(),
//                 domain_opt:self.domain_opt ,
//                 sampler_obj: ,
//                 sampler_opt: self.sampler_opt,
//                 onto_obj_fn: Opt::onto,
//                 onto_opt_fn:Obj::onto,
//                 _single_dom:true,
//             }
//         }
//         else{

//         }

//     }
// }