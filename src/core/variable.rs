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

#[derive(Clone, Copy)]
pub struct Variable<'a, Obj, Opt=Obj>
where
    Obj: Domain + Clone,
    Opt: Domain + Clone,
{
    pub name: &'a str,
    pub domain_obj: Obj,
    pub domain_opt: Opt,
    sampler_obj: fn(&Obj, &mut ThreadRng) -> Obj::TypeDom,
    sampler_opt: fn(&Opt, &mut ThreadRng) -> Opt::TypeDom,
}

impl<'a, Obj,Opt> Variable<'a, Obj,Opt>
where
    Obj: Domain + Clone,
    Opt: Domain + Clone,
{
    /// Creates a new instance of a [`Variable`].
    ///
    /// # Parameters
    ///
    /// * name : &'a str - The name of the variable, mostly used for saving, or pass a point as a keyword.
    /// * domobj : Obj - A single-[`Variable`] [`Domain`] of the [`Objective`] [`Domain`].
    /// * domopt : Opt - A single-[`Variable`] [`Domain`] of the [`Optimizer`] [`Domain`].
    /// * sampobj : Option<fn(&Obj, &mut ThreadRng) -> Obj::TypeDom> - An optional sampler function for the [`Objective`] [`Domain`].
    /// By default use the [`default_sampler`](Domain::default_sampler) of the [`Domain`].
    /// * sampopt : Option<fn(&Opt, &mut ThreadRng) -> Opt::TypeDom> - An optional sampler function for the [`Optimizer`] [`Domain`].
    /// By default use the [`default_sampler`](Domain::default_sampler) of the [`Domain`].
    ///
    pub fn new(
        name: &'a str,
        domobj: Obj,
        domopt: Opt,
        sampobj: Option<fn(&Obj, &mut ThreadRng) -> Obj::TypeDom>,
        sampopt: Option<fn(&Opt, &mut ThreadRng) -> Opt::TypeDom>,
    ) -> Variable<'a, Obj, Opt> {

        let samplerobj_selected = sampobj.unwrap_or_else(|| domobj.default_sampler());
        let sampleropt_selected = sampopt.unwrap_or_else(|| domopt.default_sampler());

        Variable {
            name: name,
            domain_obj: domobj,
            domain_opt: domopt,
            sampler_obj: samplerobj_selected,
            sampler_opt: sampleropt_selected,
        }
    }

    
}

impl <'a, Obj, Opt> Variable<'a, Obj, Opt>
where
    Obj: Domain + Clone,
    Opt: Domain + Clone,
{
    /// Samples a point from the **Objective** `domobj` [`Domain`].
    ///
    /// # Parameters
    ///
    /// * rng : &mut [`ThreadRng`] - A random number generator thread from [`rand`].
    ///
    /// # Return
    ///
    /// * `Obj::[`TypeDom`]` : A point random point within the `Obj` [`Domain`].
    ///
    pub fn sample_obj(&self, rng: &mut ThreadRng) -> Obj::TypeDom {
        (self.sampler_obj)(&self.domain_obj, rng)
    }

    /// Samples a point from the **Optimizer** `domopt` [`Domain`].
    ///
    /// # Parameters
    ///
    /// * rng : &mut [`ThreadRng`] - A random number generator thread from [`rand`].
    ///
    /// # Return
    ///
    /// * `Opt::[`TypeDom`]` : A point random point within the `Opt` [`Domain`].
    ///
    pub fn sample_opt(&self, rng: &mut ThreadRng) -> Opt::TypeDom {
        (self.sampler_opt)(&self.domain_opt, rng)
    }
    /// Maps an input point from the **Objective** `domobj` [`Domain`] to the **Optimizer**
    /// `domopt` [`Domain`].
    /// See [`Onto`] for more info.
    ///
    /// # Parameters
    ///
    /// * item : &Obj::TypeDom - A point from the `Obj` [`Domain`] (Objective).
    ///
    /// # Return
    ///
    /// * Result<Opt::TypeDom, DomainError>
    ///
    /// The function return the `item` mapped into the `Opt` [`Domain`].
    /// If an error occurs it returns a [`DomainError`].
    ///
    /// # Errors
    ///
    /// * [`DomainError`]
    /// If the `item` is not into the `Obj` [`Domain`],
    /// or if the mapped `item` is not into the `Obj` [`Domain`].
    ///
    pub fn onto_opt(&self, item: &Obj::TypeDom) -> Result<Opt::TypeDom, DomainError> 
    where
        Obj:Onto<Opt>
    {
        self.domain_obj.onto(item, &self.domain_opt)
    }

    /// Maps an input point from the **Optimizer** `domopt` [`Domain`] to the **Objective**
    /// `domobj` [`Domain`].
    /// See [`Onto`] for more info.
    ///
    /// # Parameters
    ///
    /// * item : &Opt::TypeDom - A point from the `Opt` [`Domain`] (Optimizer).
    ///
    /// # Return
    ///
    /// * Result<Obj::TypeDom, DomainError>
    ///
    /// The function return the `item` mapped into the `Obj` [`Domain`].
    /// If an error occurs it returns a [`DomainError`].
    ///
    /// # Errors
    ///
    /// * [`DomainError`]
    /// If the `item` is not into the `Opt` [`Domain`],
    /// or if the mapped `item` is not into the `Obj` [`Domain`].
    ///
    pub fn onto_obj(&self, item: &Opt::TypeDom) -> Result<Obj::TypeDom, DomainError> 
    where
        Opt:Onto<Obj>
    {
        self.domain_opt.onto(item, &self.domain_obj)
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
/// let random_obj = variable.sample_obj(&mut rng);
/// assert!(variable.domain_obj.is_in(&random_obj));
/// 
/// // Default Optimizer part (Clone of domain_obj)
/// assert_eq!(variable.domain_opt.lower(),0.0);
/// assert_eq!(variable.domain_opt.upper(),10.0);
/// // Using default sampler for optimizer's domain.
/// let random_opt = variable.sample_opt(&mut rng);
/// assert!(variable.domain_opt.is_in(&random_opt));
/// ```
///
/// - Create a [`Variable`] with the [`Domain`] of the [`Objective`] function
/// and the [`Domain`] of the [`Optimizer`] algorithm
///
/// ```
/// use tantale::core::domain::{Real, Domain, DomainBounded};
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
/// let random_obj = variable.sample_obj(&mut rng);
/// assert!(variable.domain_obj.is_in(&random_obj));
/// 
/// // Optimizer part (Defined above)
/// assert_eq!(variable.domain_opt.lower(),0.0);
/// assert_eq!(variable.domain_opt.upper(),1.0);
/// // Using default sampler for optimizer's domain.
/// let random_opt = variable.sample_opt(&mut rng);
/// assert!(variable.domain_opt.is_in(&random_opt));
/// ```
///
/// - Create a [`Variable`] with the [`Domain`] of the [`Objective`] function
/// and a sampler.
///
/// ```
/// use tantale::core::domain::{Real, Domain, DomainBounded};
/// use tantale::core::sampler::uniform_real;
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
/// let random_obj = variable.sample_obj(&mut rng);
/// assert!(variable.domain_obj.is_in(&random_obj));
/// 
/// // Default Optimizer part (Clone of domain_obj)
/// assert_eq!(variable.domain_opt.lower(),80.0);
/// assert_eq!(variable.domain_opt.upper(),100.0);
/// // Using default sampler for optimizer's domain.
/// let random_opt = variable.sample_opt(&mut rng);
/// assert!(variable.domain_opt.is_in(&random_opt));
/// ```
/// 
/// - All possibilities
/// ```
/// use tantale::core::domain::{Real, Int, Domain, DomainBounded};
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
        $crate::core::variable::Variable::new($name,$domobj,$domopt,Some($sampobj),Some($sampopt))
    );
    // Solely defining objective sampler
    ($name:literal ; obj | $domobj:expr => $sampobj:expr ; opt | $domopt:expr) => (
        $crate::core::variable::Variable::new($name,$domobj,$domopt,Some($sampobj),None)
    );
    // Solely defining optimizer sampler
    ($name:literal ; obj | $domobj:expr ; opt | $domopt:expr => $sampopt:expr) => (
        $crate::core::variable::Variable::new($name,$domobj,$domopt,None,Some($sampopt))
    );
    // No sampler
    ($name:literal ; obj | $domobj:expr ; opt | $domopt:expr) => (
        $crate::core::variable::Variable::new($name,$domobj,$domopt,None,None)
    );

    // Solely defining objective domain
    // Defining both samplers
    ($name:literal ; obj | $domobj:expr => $sampobj:expr ; opt | => $sampopt:expr) => ({
        let domopt = $domobj.clone();
        $crate::core::variable::Variable::new($name,$domobj,domopt,Some($sampobj),Some($sampopt))
    });
    // Solely defining objective sampler
    ($name:literal ; obj | $domobj:expr => $sampobj:expr) => ({
        let domopt = $domobj.clone();
        $crate::core::variable::Variable::new($name,$domobj,domopt,Some($sampobj),None)
    });
    // Solely defining optimizer sampler
    ($name:literal ;  obj | $domobj:expr ; opt | => $sampopt:expr) => ({
        let domopt = $domobj.clone();
        $crate::core::variable::Variable::new($name,$domobj,domopt,None,Some($sampopt))
    });
    // No sampler
    ($name:literal ; obj | $domobj:expr) => ({
        let domopt = $domobj.clone();
        $crate::core::variable::Variable::new($name,$domobj,domopt,None,None)
    });
}