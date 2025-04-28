#[cfg(doc)]
use crate::objective::Objective;
#[cfg(doc)]
use crate::optimizer::Optimizer;

/// # Variable
/// This crate describes what a variable is.
/// Most of the domains implements the [`Domain`] type trait [TypeDom](Domain::TypeDom).
/// It gives the type of a point within this domain.
/// [`Domains`] are use in [`Variable`] to define the type of the variable for the input type of
/// that variable within the [`Objective`] function, and the input type of the [`Optimizer`].
///
use crate::domain::{
    bool::Bool,
    bounded::{Int, Nat, Real},
    cat::Cat,
    onto::Onto,
    unit::Unit,
    Domain,
};

use rand::prelude::ThreadRng;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use super::DomainError;

/// Describes a [`Variable`] with an [`Objective`] [`Domain`]  and an [`Optimizer`] [`Domain`].
///
#[derive(Clone)]
pub struct Variable<'a, Obj, Opt = Obj>
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

impl<'a, Obj> Variable<'a, Obj>
where
    Obj: Domain + Clone + Display + Debug,
{
    //// Creates a new instance of a [`Variable`] when only the [`Objective`] [`Domain`] is defined.
    ///
    /// # Parameters
    ///
    /// * `name` : `&'a str` - Name of the variable.
    /// The name of the variable, mostly used for saving, or pass a point as a keyword.
    /// * `domobj` : [`Rc`]`<Obj>` - Accessible via the method [`domain_obj()`](Variable::domain_obj).
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
    ) -> Variable<'a, Obj> {
        let samplerobj_selected = sampobj.unwrap_or(Obj::sample);
        let sampleropt_selected = sampopt.unwrap_or(Obj::sample);

        Variable {
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

impl<'a, Obj, Opt> Variable<'a, Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug + Onto<Opt>,
    Opt: Domain + Clone + Display + Debug + Onto<Obj>,
{
    //// Creates a new instance of a [`Variable`] when the [`Objective`] and [`Optimizer`] [`Domain`]s are defined.
    ///
    /// # Parameters
    ///
    /// * `name` : `&'a str` - Name of the variable.
    /// The name of the variable, mostly used for saving, or pass a point as a keyword.
    /// * `domobj` : [`Rc`]`<Obj>` - Accessible via the method [`domain_obj()`](Variable::domain_obj).
    /// The [`Domain`] of the [`Objective`] [`Domain`].
    /// * `domopt` : [`Rc`]`<Opt>` - Accessible via the method [`domain_opt()`](Variable::domain_opt).
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
    ) -> Variable<'a, Obj, Opt> {
        let samplerobj_selected = sampobj.unwrap_or(Obj::sample);
        let sampleropt_selected = sampopt.unwrap_or(Opt::sample);

        Variable {
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

impl<'a, Obj, Opt> Variable<'a, Obj, Opt>
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
//     fn into_obj_base(self)->Variable<'a, BaseDom, Opt>;
//     fn into_opt_base(self)->Variable<'a, Obj, BaseDom>;
//     fn into_single_base(self)->Variable<'a, BaseDom>;
// }

// impl IntoBase for Variable<'a, Real>
// {
//     fn into_single_base(self, wrapped:)->Variable<'a, BaseDom> {
//         let domobj = *self.domain_obj;
//         let domobj:BaseDom = domobj.into();
//         let domobj = Rc::new(domobj);

//         Variable{
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

// impl<'a,Opt> Variable<'a, Real, Opt>
// where
//     Opt: Domain + Clone + Display + Debug,
// {
//     pub fn into_obj_base(self)->Variable<'a, BaseDom, Opt>{
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
//             Variable{
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

/// Creates a [`Variable`] containing the arguments.
///
/// `var!` simplifies the definition of a [`Variable`] hiding optional parameters.
/// The full syntax to define a [`Variable`] with `var!` is as follows :
///
/// - `var!("NAME" ; obj | DOMAIN_1 => SAMPLER_1 ; opt | DOMAIN_2 => SAMPLER_2)`
///
/// with `obj` describing the part concerning the [`Domain`] of the [`Objective`] function,
/// and `opt` describing the part concerning the [`Domain`] of the [`Optimizer`] algorithm.
///
/// # Parameters
///
/// * `NAME` : `&'a str` - the  name of the variable. Used for saving.
/// * `DOMAIN_1` : `Obj` - [`Domain`] of the [`Objective`].
/// * `SAMPLER_1` : `fn(&Obj, &mut `[`ThreadRng`]`) -> Obj::`[`TypeDom`](Domain::TypeDom)` - Optional sampler function for the [`Domain`] of the [`Objective`].
/// If ignored the default sampler of the [`Domain`] is used.
/// * `DOMAIN_2` : `Opt` - Optional [`Domain`] of the [`Optimizer`].
/// If ignored within the macros `DOMAIN_1` is cloned to fill `DOMAIN_2`.
/// * `SAMPLER_2` : `fn(&Opt, &mut `[`ThreadRng`]`) -> Opt::`[`TypeDom`](Domain::TypeDom) - Optional sampler function for the [`Domain`] of the [`Optimizer`].
/// If ignored the default sampler of the [`Domain`] is used.
///
/// # Types
///
/// * Opt : [`Domain`]` + `[`Clone`]
/// * Obj : [`Domain`]` + `[`Clone`]
///  
/// # Notes
///
/// **The name is mandatory**.
/// The samplers and the [`Optimizer`]'s [`Domain`] are optionals.
///
/// # Examples
///
/// - Create a [`Variable`] with only the [`Domain`] of the [`Objective`] function
///
/// ```
/// use tantale::core::domain::{Real, Domain, DomainBounded};
/// use tantale::core::variable::Variable;
/// use tantale::var;
///
/// let domobj = Real::new(0.0, 10.0);
/// let variable = var!("a" ; obj | domobj);
///
/// assert_eq!(variable.name,"a");
///
/// let mut rng = rand::rng();
///
/// // Objective part (Defined above)
/// assert_eq!(variable.domain_obj.lower(),0.0);
/// assert_eq!(variable.domain_obj.upper(),10.0);
/// // Using default sampler for objective's domain.
/// let random_obj = (variable.sampler_obj)(&variable.domain_obj,&mut rng);
/// assert!(variable.domain_obj.is_in(&random_obj));
///
/// // Default Optimizer part (Clone of domain_obj)
/// assert_eq!(variable.domain_opt.lower(),0.0);
/// assert_eq!(variable.domain_opt.upper(),10.0);
/// // Using default sampler for optimizer's domain.
/// let random_opt = (variable.sampler_opt)(&variable.domain_opt,&mut rng);
/// assert!(variable.domain_opt.is_in(&random_opt));
/// ```
///
/// - Create a [`Variable`] with the [`Domain`] of the [`Objective`] function
/// and the [`Domain`] of the [`Optimizer`] algorithm
///
/// ```
/// use tantale::core::domain::{Real, Domain, DomainBounded};
/// use tantale::core::variable::Variable;
/// use tantale::var;
///
/// let domobj = Real::new(80.0, 100.0);
/// let domopt = Real::new(0.0, 1.0);
/// let variable = var!("a" ; obj | domobj ; opt | domopt);
///
/// assert_eq!(variable.name,"a");
///
/// let mut rng = rand::rng();
///
/// // Objective part (Defined above)
/// assert_eq!(variable.domain_obj.lower(),80.0);
/// assert_eq!(variable.domain_obj.upper(),100.0);
/// // Using default sampler for objective's domain.
/// let random_obj = (variable.sampler_obj)(&variable.domain_obj,&mut rng);
/// assert!(variable.domain_obj.is_in(&random_obj));
///
/// // Optimizer part (Defined above)
/// assert_eq!(variable.domain_opt.lower(),0.0);
/// assert_eq!(variable.domain_opt.upper(),1.0);
/// // Using default sampler for optimizer's domain.
/// let random_opt = (variable.sampler_opt)(&variable.domain_opt,&mut rng);
/// assert!(variable.domain_opt.is_in(&random_opt));
/// ```
///
/// - Create a [`Variable`] with the [`Domain`] of the [`Objective`] function
/// and a sampler.
///
/// ```
/// use tantale::core::domain::{Real, Domain, DomainBounded};
/// use tantale::core::variable::Variable;
/// use tantale::core::domain::sampler::uniform_real;
/// use tantale::var;
///
/// let domobj = Real::new(80.0, 100.0);
/// let variable = var!("a" ; obj | domobj => uniform_real);
///
/// assert_eq!(variable.name,"a");
///
/// let mut rng = rand::rng();
///
/// // Objective part (Defined above)
/// assert_eq!(variable.domain_obj.lower(),80.0);
/// assert_eq!(variable.domain_obj.upper(),100.0);
/// // Using given sampler for objective's domain.
/// let random_obj = (variable.sampler_obj)(&variable.domain_obj,&mut rng);
/// assert!(variable.domain_obj.is_in(&random_obj));
///
/// // Default Optimizer part (Clone of domain_obj)
/// assert_eq!(variable.domain_opt.lower(),80.0);
/// assert_eq!(variable.domain_opt.upper(),100.0);
/// // Using default sampler for optimizer's domain.
/// let random_opt = (variable.sampler_opt)(&variable.domain_opt,&mut rng);
/// assert!(variable.domain_opt.is_in(&random_opt));
/// ```
///
/// - All possibilities
/// ```
/// use tantale::core::domain::{Real, Int, Domain, DomainBounded};
/// /// use tantale::core::variable::Variable;
/// use tantale::core::domain::sampler::{uniform_real,uniform_int};
/// use tantale::var;
///
/// let domobj = Int::new(0, 100);
/// let variable = var!("a" ; obj | domobj);
///
/// let domobj = Int::new(0, 100);
/// let variable = var!("a" ; obj | domobj => uniform_int);
///
/// let domobj = Int::new(0, 100);
/// let variable = var!("a" ; obj | domobj => uniform_int ; opt | => uniform_int);
///
/// let domobj = Int::new(0, 100);
/// let variable = var!("a" ; obj | domobj ; opt | => uniform_int);
///
/// let domobj = Int::new(0, 100);
/// let domopt = Real::new(0.0, 1.0);
/// let variable = var!("a" ; obj | domobj ; opt | domopt);
///
/// let domobj = Int::new(0, 100);
/// let domopt = Real::new(0.0, 1.0);
/// let variable = var!("a" ; obj | domobj => uniform_int ; opt | domopt => uniform_real);
///
/// let domobj = Int::new(0, 100);
/// let domopt = Real::new(0.0, 1.0);
/// let variable = var!("a" ; obj | domobj => uniform_int ; opt | domopt);
///
/// let domobj = Int::new(0, 100);
/// let domopt = Real::new(0.0, 1.0);
/// let variable = var!("a" ; obj | domobj ; opt | domopt => uniform_real);
/// ```
#[macro_export]
macro_rules! var {
    // Defining both objective and optimizer domains
    // Defining both samplers
    ($name:literal ; obj | $domobj:expr => $sampobj:expr ; opt | $domopt:expr => $sampopt:expr) => {{
        $crate::variable::Variable::new_double(
            $name,
            std::rc::Rc::new($domobj),
            std::rc::Rc::new($domopt),
            Some($sampobj),
            Some($sampopt),
        )
    }};
    // Solely defining objective sampler
    ($name:literal ; obj | $domobj:expr => $sampobj:expr ; opt | $domopt:expr) => {{
        $crate::variable::Variable::new_double(
            $name,
            std::rc::Rc::new($domobj),
            std::rc::Rc::new($domopt),
            Some($sampobj),
            None,
        )
    }};
    // Solely defining optimizer sampler
    ($name:literal ; obj | $domobj:expr ; opt | $domopt:expr => $sampopt:expr) => {{
        $crate::variable::Variable::new_double(
            $name,
            std::rc::Rc::new($domobj),
            std::rc::Rc::new($domopt),
            None,
            Some($sampopt),
        )
    }};
    // No sampler
    ($name:literal ; obj | $domobj:expr ; opt | $domopt:expr) => {{
        $crate::variable::Variable::new_double(
            $name,
            std::rc::Rc::new($domobj),
            std::rc::Rc::new($domopt),
            None,
            None,
        )
    }};

    // Solely defining objective domain
    // Defining both samplers
    ($name:literal ; obj | $domobj:expr => $sampobj:expr ; opt | => $sampopt:expr) => {{
        $crate::variable::Variable::new_single(
            $name,
            std::rc::Rc::new($domobj),
            Some($sampobj),
            Some($sampopt),
        )
    }};
    // Solely defining objective sampler
    ($name:literal ; obj | $domobj:expr => $sampobj:expr) => {{
        $crate::variable::Variable::new_single(
            $name,
            std::rc::Rc::new($domobj),
            Some($sampobj),
            None,
        )
    }};
    // Solely defining optimizer sampler
    ($name:literal ;  obj | $domobj:expr ; opt | => $sampopt:expr) => {{
        $crate::variable::Variable::new_single(
            $name,
            std::rc::Rc::new($domobj),
            None,
            Some($sampopt),
        )
    }};
    // No sampler
    ($name:literal ; obj | $domobj:expr) => {{
        $crate::variable::Variable::new_single($name, std::rc::Rc::new($domobj), None, None)
    }};
}
