#[cfg(doc)]
use crate::core::objective::Objective;
#[cfg(doc)]
use crate::core::optimizer::Optimizer;

/// # Variable
/// This crate describes what a variable is.
/// Most of the domains implements the [`Domain`] type trait [TypeDom](Domain::TypeDom).
/// It gives the type of a point within this domain.
/// [`Domains`] are use in [`Variable`] to define the type of the variable for the input type of
/// that variable within the [`Objective`] function, and the input type of the [`Optimizer`].
///
use crate::core::domain::Domain;
use crate::core::errors::DomainError;
use crate::core::onto::Onto;

use rand::prelude::ThreadRng;
use std::rc::Rc;

pub trait Variable<'a>
{
    type TypeObj: Domain + Clone;
    type TypeOpt: Domain + Clone;

    /// Creates a new instance of a [`VariableDouble`].
    ///
    /// # Parameters
    ///
    /// * `name` : `&'a str` - Accessible via the method [`name()`](Variable::name) .The name of the variable, mostly used for saving, or pass a point as a keyword.
    /// * `domobj` : `Obj` - Accessible via the method [`domain_obj()`](Variable::domain_obj) .A single-[`Variable`] [`Domain`] of the [`Objective`] [`Domain`].
    /// * `domopt` : `Opt` - Accessible via the method [`domain_opt()`](Variable::domain_opt) .A single-[`Variable`] [`Domain`] of the [`Optimizer`] [`Domain`].
    /// * `sampobj` : [`Option`]`<fn(&Obj, &mut `[`ThreadRng`]`) -> Obj::`[`TypeDom`](Domain::TypeDom)`>` - An optional sampler function for the [`Objective`] [`Domain`].
    /// By default use the [`default_sampler`](Domain::default_sampler) of the [`Domain`].
    /// * sampopt : Option<fn(&Opt, &mut ThreadRng) -> Opt::TypeDom> - An optional sampler function for the [`Optimizer`] [`Domain`].
    /// By default use the [`default_sampler`](Domain::default_sampler) of the [`Domain`].
    ///
    fn new(
        name: &'a str,
        domobj: Rc<Self::TypeObj>,
        domopt: Option<Rc<Self::TypeOpt>>,
        sampobj: Option<fn(&Self::TypeObj, &mut ThreadRng) -> <Self::TypeObj as Domain>::TypeDom>,
        sampopt: Option<fn(&Self::TypeOpt, &mut ThreadRng) -> <Self::TypeOpt as Domain>::TypeDom>,
    ) -> Self;
    /// Returns the name of the [`Variable`].
    fn name(&self) -> &'a str;
    /// Returns the domain of the [`Objective`] function.
    fn domain_obj(&self) -> Rc<Self::TypeObj>;
    /// Returns the domain of the [`Optimizer`] algorithm.
    fn domain_opt(&self) -> Rc<Self::TypeOpt>;
    /// Samples a point from the **Objective** `domobj` [`Domain`].
    ///
    /// # Parameters
    ///
    /// * rng : `&mut `[`ThreadRng`] - A random number generator thread from [`rand`].
    ///
    /// # Return
    ///
    /// * `Obj::`[`TypeDom`](Domain::TypeDom) : A random point within the `Obj` [`Domain`].
    ///
    fn sample_obj(&self, rng: &mut ThreadRng) -> <Self::TypeObj as Domain>::TypeDom;

    /// Samples a point from the **Optimizer** `domopt` [`Domain`].
    ///
    /// # Parameters
    ///
    /// * `rng` : `&mut `[`ThreadRng`] - A random number generator thread from [`rand`].
    ///
    /// # Return
    ///
    /// * `Opt::`[`TypeDom`](Domain::TypeDom) : A random point within the `Opt` [`Domain`].
    ///
    fn sample_opt(&self, rng: &mut ThreadRng) -> <Self::TypeOpt as Domain>::TypeDom;
    /// Replicates a given [`Variable`] to create a new one with a new given name.
    ///
    /// # Parameters
    ///
    /// * `name` : `&'a str` - The name of the new [`Variable`].
    /// * `item` : [`Variable`]`<a, Obj, Opt>`
    ///
    /// # Return
    ///
    /// * [`Variable`]`<'b, Obj, Opt>`
    /// 
    fn replicate(&self,name:&'a str) -> Self;
    /// Maps an input point from the [`Objective`] `domopt` [`Domain`] to the [`Optimizer`]
    /// `domopt` [`Domain`].
    /// See [`Onto`] for more info.
    ///
    /// # Parameters
    ///
    /// * `item` : `&Obj::`[`TypeDom`](Domain::TypeDom) - A point from the `Obj` [`Domain`] (Objective).
    ///
    /// # Return
    ///
    /// * [`Result`]`<Opt::`[`TypeDom`](Domain::TypeDom)`, `[`DomainError`]`>`
    ///
    /// The function return the `item` mapped into the `Opt` [`Domain`].
    /// If an error occurs it returns a [`DomainError`].
    ///
    /// # Types
    /// 
    /// * `Obj` : [`Domain`] + [`Clone`] + [`Onto`]`<Opt>`
    /// 
    /// # Errors
    ///
    /// * [`DomainError`]
    /// If the `item` is not into the `Obj` [`Domain`],
    /// or if the mapped `item` is not into the `Obj` [`Domain`].
    ///
    fn onto_opt(&self, item: &<Self::TypeObj as Domain>::TypeDom) -> Result<<Self::TypeOpt as Domain>::TypeDom, DomainError>;

    /// Maps an input point from the [`Optimizer`] `domopt` [`Domain`] to the [`Objective`]
    /// `domobj` [`Domain`].
    /// See [`Onto`] for more info.
    ///
    /// # Parameters
    ///
    /// * `item` : `&Opt::`[`TypeDom`](Domain::TypeDom) - A point from the `Opt` [`Domain`] (Optimizer).
    ///
    /// # Return
    ///
    /// * [`Result`]`<Obj::`[`TypeDom`](Domain::TypeDom)`, `[`DomainError`]`>`
    ///
    /// The function return the `item` mapped into the `Obj` [`Domain`].
    /// If an error occurs it returns a [`DomainError`].
    ///
    /// # Types
    /// 
    /// * `Opt` : [`Domain`] + [`Clone`] + [`Onto`]`<Obj>`
    /// 
    /// # Errors
    ///
    /// * [`DomainError`]
    /// If the `item` is not into the `Opt` [`Domain`],
    /// or if the mapped `item` is not into the `Obj` [`Domain`].
    ///
    fn onto_obj(&self, item: &<Self::TypeOpt as Domain>::TypeDom) -> Result<<Self::TypeObj as Domain>::TypeDom, DomainError>;
}

#[derive(Clone)]
pub struct VariableDouble<'a, Obj, Opt>
where
    Obj: Domain + Clone,
    Opt: Domain + Clone,
{
    name: &'a str,
    domain_obj: Rc<Obj>,
    domain_opt: Rc<Opt>,
    sampler_obj: fn(&Obj, &mut ThreadRng) -> Obj::TypeDom,
    sampler_opt: fn(&Opt, &mut ThreadRng) -> Opt::TypeDom,
}

impl <'a, Obj, Opt> Variable<'a> for VariableDouble<'a, Obj, Opt>
where
    Obj: Domain + Clone + Onto<Opt>,
    Opt: Domain + Clone + Onto<Obj>,
{
    type TypeObj = Obj;
    type TypeOpt = Opt;

    /// Creates a new instance of a [`VariableDouble`].
    ///
    /// # Parameters
    ///
    /// * `name` : `&'a str` - The name of the variable, mostly used for saving, or pass a point as a keyword.
    /// * `domobj` : `Obj` - A single-[`Variable`] [`Domain`] of the [`Objective`] [`Domain`].
    /// * `domopt` : `Opt` - A single-[`Variable`] [`Domain`] of the [`Optimizer`] [`Domain`].
    /// * `sampobj` : [`Option`]`<fn(&Obj, &mut `[`ThreadRng`]`) -> Obj::`[`TypeDom`](Domain::TypeDom)`>` - An optional sampler function for the [`Objective`] [`Domain`].
    /// By default use the [`default_sampler`](Domain::default_sampler) of the [`Domain`].
    /// * sampopt : Option<fn(&Opt, &mut ThreadRng) -> Opt::TypeDom> - An optional sampler function for the [`Optimizer`] [`Domain`].
    /// By default use the [`default_sampler`](Domain::default_sampler) of the [`Domain`].
    ///
    fn new(
        name: &'a str,
        domobj: Rc<Self::TypeObj>,
        domopt: Option<Rc<Self::TypeOpt>>,
        sampobj: Option<fn(&Self::TypeObj, &mut ThreadRng) -> <Self::TypeObj as Domain>::TypeDom>,
        sampopt: Option<fn(&Self::TypeOpt, &mut ThreadRng) -> <Self::TypeOpt as Domain>::TypeDom>,
    ) -> VariableDouble<'a,Self::TypeObj, Self::TypeOpt> {

        let dopt = domopt.expect("For VariableDouble, domain`domopt` cannot be None" );
        let samplerobj_selected = sampobj.unwrap_or_else(|| domobj.default_sampler());
        let sampleropt_selected = sampopt.unwrap_or_else(|| dopt.default_sampler());

        VariableDouble {
            name: name,
            domain_obj: domobj,
            domain_opt: dopt,
            sampler_obj: samplerobj_selected,
            sampler_opt: sampleropt_selected,
        }
    }
    fn name(&self) -> &'a str {
        self.name
    }
    fn domain_obj(&self) -> Rc<Self::TypeObj> {
        self.domain_obj.clone()
    }
    fn domain_opt(&self) -> Rc<Self::TypeOpt> {
        self.domain_opt.clone()
    }
    fn sample_obj(&self, rng: &mut ThreadRng) -> Obj::TypeDom {
        (self.sampler_obj)(&self.domain_obj, rng)
    }
    fn sample_opt(&self, rng: &mut ThreadRng) -> Opt::TypeDom {
        (self.sampler_opt)(&self.domain_opt, rng)
    }
    fn replicate(&self, name:&'a str) -> VariableDouble<'a,Obj,Opt>
    {
        let domobj = Rc::clone(&self.domain_obj);
        let domopt = Rc::clone(&self.domain_opt);
        <Self as Variable>::new(name, domobj, Some(domopt), Some(self.sampler_obj), Some(self.sampler_opt))
    }
    fn onto_opt(&self, item: &Obj::TypeDom) -> Result<Opt::TypeDom, DomainError> 
    {
        self.domain_obj.onto(item, &self.domain_opt)
    }
    fn onto_obj(&self, item: &Opt::TypeDom) -> Result<Obj::TypeDom, DomainError> 
    {
        self.domain_opt.onto(item, &self.domain_obj)
    }
}

#[derive(Clone)]
pub struct VariableSingle<'a, Obj>
where
    Obj: Domain + Clone,
{
    name: &'a str,
    domain_obj: Rc<Obj>,
    sampler_obj: fn(&Obj, &mut ThreadRng) -> Obj::TypeDom,
    sampler_opt: fn(&Obj, &mut ThreadRng) -> Obj::TypeDom,
}

impl <'a, Obj> Variable<'a> for VariableSingle<'a, Obj>
where
    Obj: Domain + Clone,
    Obj::TypeDom : Clone,
{
    type TypeObj = Obj;
    type TypeOpt = Obj;
    /// Creates a new instance of a [`VariableSingle`].
    ///
    /// # Parameters
    ///
    /// * `name` : `&'a str` - The name of the variable, mostly used for saving, or pass a point as a keyword.
    /// * `domobj` : `Obj` - A single-[`Variable`] [`Domain`] of the [`Objective`] [`Domain`].
    /// * `sampobj` : [`Option`]`<fn(&Obj, &mut `[`ThreadRng`]`) -> Obj::`[`TypeDom`](Domain::TypeDom)`>` - An optional sampler function for the [`Objective`] [`Domain`].
    /// By default use the [`default_sampler`](Domain::default_sampler) of the [`Domain`].
    /// * sampopt : Option<fn(&Opt, &mut ThreadRng) -> Opt::TypeDom> - An optional sampler function for the [`Optimizer`] [`Domain`].
    /// By default use the [`default_sampler`](Domain::default_sampler) of the [`Domain`].
    ///
    fn new(
        name: &'a str,
        domobj: Rc<Self::TypeObj>,
        domopt: Option<Rc<Self::TypeObj>>,
        sampobj: Option<fn(&Self::TypeObj, &mut ThreadRng) -> <Self::TypeObj as Domain>::TypeDom>,
        sampopt: Option<fn(&Self::TypeObj, &mut ThreadRng) -> <Self::TypeObj as Domain>::TypeDom>,
    ) -> VariableSingle<'a,Self::TypeObj> {


        match domopt{
            Some(_e) => panic!("For VariableSingle, `domopt` sould be None"),
            None => {
                let samplerobj_selected = sampobj.unwrap_or_else(|| domobj.default_sampler());
                let sampleropt_selected = sampopt.unwrap_or_else(|| domobj.default_sampler());

                VariableSingle {
                    name: name,
                    domain_obj: domobj,
                    sampler_obj: samplerobj_selected,
                    sampler_opt: sampleropt_selected,
                }
            }
        }
    }
    fn name(&self) -> &'a str {
        self.name
    }
    fn domain_obj(&self) -> Rc<Self::TypeObj> {
        self.domain_obj.clone()
    }
    fn domain_opt(&self) -> Rc<Self::TypeOpt> {
        self.domain_obj.clone()
    }
    fn sample_obj(&self, rng: &mut ThreadRng) -> Obj::TypeDom {
        (self.sampler_obj)(&self.domain_obj, rng)
    }
    fn sample_opt(&self, rng: &mut ThreadRng) -> Obj::TypeDom {
        (self.sampler_opt)(&self.domain_obj, rng)
    }
    fn replicate(&self, name:&'a str) -> VariableSingle<'a,Obj>
    {
        let domobj = Rc::clone(&self.domain_obj);
        <Self as Variable>::new(name, domobj, None,Some(self.sampler_obj), Some(self.sampler_opt))
    }
    fn onto_opt(&self, item: &Obj::TypeDom) -> Result<Obj::TypeDom, DomainError>
    {
        Ok(item.clone())
    }
    fn onto_obj(&self, item: &Obj::TypeDom) -> Result<Obj::TypeDom, DomainError> 
    {
        Ok(item.clone())
    }
}

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
/// assert_eq!(variable.name(),"a");
///
/// let mut rng = rand::rng();
/// 
/// // Objective part (Defined above) 
/// assert_eq!(variable.domain_obj().lower(),0.0);
/// assert_eq!(variable.domain_obj().upper(),10.0);
/// // Using default sampler for objective's domain.
/// let random_obj = variable.sample_obj(&mut rng);
/// assert!(variable.domain_obj().is_in(&random_obj));
/// 
/// // Default Optimizer part (Clone of domain_obj)
/// assert_eq!(variable.domain_opt().lower(),0.0);
/// assert_eq!(variable.domain_opt().upper(),10.0);
/// // Using default sampler for optimizer's domain.
/// let random_opt = variable.sample_opt(&mut rng);
/// assert!(variable.domain_opt().is_in(&random_opt));
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
/// assert_eq!(variable.name(),"a");
///
/// let mut rng = rand::rng();
/// 
/// // Objective part (Defined above) 
/// assert_eq!(variable.domain_obj().lower(),80.0);
/// assert_eq!(variable.domain_obj().upper(),100.0);
/// // Using default sampler for objective's domain.
/// let random_obj = variable.sample_obj(&mut rng);
/// assert!(variable.domain_obj().is_in(&random_obj));
/// 
/// // Optimizer part (Defined above)
/// assert_eq!(variable.domain_opt().lower(),0.0);
/// assert_eq!(variable.domain_opt().upper(),1.0);
/// // Using default sampler for optimizer's domain.
/// let random_opt = variable.sample_opt(&mut rng);
/// assert!(variable.domain_opt().is_in(&random_opt));
/// ```
///
/// - Create a [`Variable`] with the [`Domain`] of the [`Objective`] function
/// and a sampler.
///
/// ```
/// use tantale::core::domain::{Real, Domain, DomainBounded};
/// use tantale::core::variable::Variable;
/// use tantale::core::sampler::uniform_real;
/// use tantale::var;
/// 
/// let domobj = Real::new(80.0, 100.0);
/// let variable = var!("a" ; obj | domobj => uniform_real);
/// 
/// assert_eq!(variable.name(),"a");
///
/// let mut rng = rand::rng();
/// 
/// // Objective part (Defined above) 
/// assert_eq!(variable.domain_obj().lower(),80.0);
/// assert_eq!(variable.domain_obj().upper(),100.0);
/// // Using given sampler for objective's domain.
/// let random_obj = variable.sample_obj(&mut rng);
/// assert!(variable.domain_obj().is_in(&random_obj));
/// 
/// // Default Optimizer part (Clone of domain_obj)
/// assert_eq!(variable.domain_opt().lower(),80.0);
/// assert_eq!(variable.domain_opt().upper(),100.0);
/// // Using default sampler for optimizer's domain.
/// let random_opt = variable.sample_opt(&mut rng);
/// assert!(variable.domain_opt().is_in(&random_opt));
/// ```
/// 
/// - All possibilities
/// ```
/// use tantale::core::domain::{Real, Int, Domain, DomainBounded};
/// /// use tantale::core::variable::Variable;
/// use tantale::core::sampler::{uniform_real,uniform_int};
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
    ($name:literal ; obj | $domobj:expr => $sampobj:expr ; opt | $domopt:expr => $sampopt:expr) => (
        {
            use $crate::core::variable::Variable;
            $crate::core::variable::VariableDouble::new($name,std::rc::Rc::new($domobj),Some(std::rc::Rc::new($domopt)),Some($sampobj),Some($sampopt))
        }
    );
    // Solely defining objective sampler
    ($name:literal ; obj | $domobj:expr => $sampobj:expr ; opt | $domopt:expr) => (
        {
            use $crate::core::variable::Variable;
            $crate::core::variable::VariableDouble::new($name,std::rc::Rc::new($domobj),Some(std::rc::Rc::new($domopt)),Some($sampobj),None)
        }
    );
    // Solely defining optimizer sampler
    ($name:literal ; obj | $domobj:expr ; opt | $domopt:expr => $sampopt:expr) => (
        {
            use $crate::core::variable::Variable;
            $crate::core::variable::VariableDouble::new($name,std::rc::Rc::new($domobj),Some(std::rc::Rc::new($domopt)),None,Some($sampopt))
        }
    );
    // No sampler
    ($name:literal ; obj | $domobj:expr ; opt | $domopt:expr) => (
        {
            use $crate::core::variable::Variable;
            $crate::core::variable::VariableDouble::new($name,std::rc::Rc::new($domobj),Some(std::rc::Rc::new($domopt)),None,None)
        }
    );

    // Solely defining objective domain
    // Defining both samplers
    ($name:literal ; obj | $domobj:expr => $sampobj:expr ; opt | => $sampopt:expr) => ({
        use $crate::core::variable::Variable;
        let domobj = std::rc::Rc::new($domobj);
        let domopt = std::rc::Rc::clone(&domobj);
        $crate::core::variable::VariableSingle::new($name,domobj,None,Some($sampobj),Some($sampopt))
    });
    // Solely defining objective sampler
    ($name:literal ; obj | $domobj:expr => $sampobj:expr) => ({
        use $crate::core::variable::Variable;
        let domobj = std::rc::Rc::new($domobj);
        let domopt = std::rc::Rc::clone(&domobj);
        $crate::core::variable::VariableSingle::new($name,domobj,None,Some($sampobj),None)
    });
    // Solely defining optimizer sampler
    ($name:literal ;  obj | $domobj:expr ; opt | => $sampopt:expr) => ({
        use $crate::core::variable::Variable;
        let domobj = std::rc::Rc::new($domobj);
        let domopt = std::rc::Rc::clone(&domobj);
        $crate::core::variable::VariableSingle::new($name,domobj,None,None,Some($sampopt))
    });
    // No sampler
    ($name:literal ; obj | $domobj:expr) => ({
        use $crate::core::variable::Variable;
        let domobj = std::rc::Rc::new($domobj);
        let domopt = std::rc::Rc::clone(&domobj);
        $crate::core::variable::VariableSingle::new($name,domobj,None,None,None)
    });
}