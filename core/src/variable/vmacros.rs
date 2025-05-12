/// Creates a [`Var`] containing the arguments.
///
/// `var!` simplifies the definition of a [`Var`] hiding optional parameters.
/// The full syntax to define a [`Var`] with `var!` is as follows :
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
/// - Create a [`Var`] with only the [`Domain`] of the [`Objective`] function
///
/// ```
/// use tantale::core::domain::{Real, Domain, DomainBounded};
/// use tantale::core::variable::Var;
/// use tantale::core::var;
///
/// let domobj = Real::new(0.0, 10.0);
/// let variable = var!("a" ; obj | domobj);
///
/// assert_eq!(variable.name,("a", None));
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
/// - Create a [`Var`] with the [`Domain`] of the [`Objective`] function
/// and the [`Domain`] of the [`Optimizer`] algorithm
///
/// ```
/// use tantale::core::domain::{Real, Domain, DomainBounded};
/// use tantale::core::variable::Var;
/// use tantale::core::var;
///
/// let domobj = Real::new(80.0, 100.0);
/// let domopt = Real::new(0.0, 1.0);
/// let variable = var!("a" ; obj | domobj ; opt | domopt);
///
/// assert_eq!(variable.name,("a", None));
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
/// - Create a [`Var`] with the [`Domain`] of the [`Objective`] function
/// and a sampler.
///
/// ```
/// use tantale::core::domain::{Real, Domain, DomainBounded};
/// use tantale::core::variable::Var;
/// use tantale::core::domain::sampler::uniform_real;
/// use tantale::core::var;
///
/// let domobj = Real::new(80.0, 100.0);
/// let variable = var!("a" ; obj | domobj => uniform_real);
///
/// assert_eq!(variable.name,("a", None));
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
/// use tantale::core::variable::Var;
/// use tantale::core::domain::sampler::{uniform_real,uniform_int};
/// use tantale::core::var;
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
        $crate::variable::Var::new_double(
            $name,
            std::rc::Rc::new($domobj),
            std::rc::Rc::new($domopt),
            Some($sampobj),
            Some($sampopt),
        )
    }};
    // Solely defining objective sampler
    ($name:literal ; obj | $domobj:expr => $sampobj:expr ; opt | $domopt:expr) => {{
        $crate::variable::Var::new_double(
            $name,
            std::rc::Rc::new($domobj),
            std::rc::Rc::new($domopt),
            Some($sampobj),
            None,
        )
    }};
    // Solely defining optimizer sampler
    ($name:literal ; obj | $domobj:expr ; opt | $domopt:expr => $sampopt:expr) => {{
        $crate::variable::Var::new_double(
            $name,
            std::rc::Rc::new($domobj),
            std::rc::Rc::new($domopt),
            None,
            Some($sampopt),
        )
    }};
    // No sampler
    ($name:literal ; obj | $domobj:expr ; opt | $domopt:expr) => {{
        $crate::variable::Var::new_double(
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
        $crate::variable::Var::new_single(
            $name,
            std::rc::Rc::new($domobj),
            Some($sampobj),
            Some($sampopt),
        )
    }};
    // Solely defining objective sampler
    ($name:literal ; obj | $domobj:expr => $sampobj:expr) => {{
        $crate::variable::Var::new_single($name, std::rc::Rc::new($domobj), Some($sampobj), None)
    }};
    // Solely defining optimizer sampler
    ($name:literal ;  obj | $domobj:expr ; opt | => $sampopt:expr) => {{
        $crate::variable::Var::new_single($name, std::rc::Rc::new($domobj), None, Some($sampopt))
    }};
    // No sampler
    ($name:literal ; obj | $domobj:expr) => {{
        $crate::variable::Var::new_single($name, std::rc::Rc::new($domobj), None, None)
    }};
}
