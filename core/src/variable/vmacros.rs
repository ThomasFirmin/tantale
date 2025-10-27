/// Creates a [`Var`](tantale::core::Var) containing the arguments.
///
/// `var!` simplifies the definition of a [`Var`](tantale::core::Var) hiding optional parameters.
/// The full syntax to define a [`Var`](tantale::core::Var) with `var!` is as follows :
///
/// - `var!("NAME" ; DOMAIN_1 => SAMPLER_1 ; DOMAIN_2 => SAMPLER_2)`
///
/// with `obj` describing the part concerning the [`Domain`](tantale::core::Domain) of the [`Objective`](tantale::core::Objective) function,
/// and `opt` describing the part concerning the [`Domain`](tantale::core::Domain) of the [`Optimizer`](tantale::core::Optimizer) algorithm.
///
/// # Parameters
///
/// * `NAME` : `&'a str` - the  name of the variable. Used for saving.
/// * `DOMAIN_1` : `Obj` - [`Domain`](tantale::core::Domain) of the [`Objective`](tantale::core::Objective).
/// * `SAMPLER_1` : `fn(&Obj, &mut `[`ThreadRng`](rand::prelude::ThreadRng)`) -> Obj::`[`TypeDom`](Domain::TypeDom) -
///   Optional sampler function for the [`Domain`](tantale::core::Domain) of the [`Objective`](tantale::core::Objective).
///   If ignored the default sampler of the [`Domain`](tantale::core::Domain) is used.
/// * `DOMAIN_2` : `Opt` - Optional [`Domain`](tantale::core::Domain) of the [`Optimizer`](tantale::core::Domain).
///   If ignored within the macros `DOMAIN_1` is cloned to fill `DOMAIN_2`.
/// * `SAMPLER_2` : `fn(&Opt, &mut `[`ThreadRng`](rand::prelude::ThreadRng)`) -> Opt::`[`TypeDom`](Domain::TypeDom) -
///   Optional sampler function for the [`Domain`](tantale::core::Domain) of the [`Optimizer`](tantale::core::Optimizer).
///   If ignored the default sampler of the [`Domain`](tantale::core::Domain) is used.
///
/// # Types
///
/// * Opt : [`Domain`](tantale::core::Domain)` + `[`Clone`]
/// * Obj : [`Domain`](tantale::core::Domain)` + `[`Clone`]
///  
/// # Notes
///
/// **The name is mandatory**.
/// The samplers and the [`Optimizer`](tantale::core::Optimizer)'s [`Domain`](tantale::core::Domain) are optionals.
///
/// # Examples
///
/// - Create a [`Var`](tantale::core::Var) with only the [`Domain`](tantale::core::Domain) of the [`Objective`](tantale::core::Objective) function
///
/// ```
/// use tantale::core::domain::{Real, Domain, DomainBounded};
/// use tantale::core::variable::Var;
/// use tantale::core::var;
///
/// let domobj = Real::new(0.0, 10.0);
/// let variable = var!("a" ; domobj);
///
/// assert_eq!(variable.get_name(),("a", None));
///
/// let mut rng = rand::rng();
///
/// // Objective part (Defined above)
/// assert_eq!(variable.get_domain_obj().lower(),0.0);
/// assert_eq!(variable.get_domain_obj().upper(),10.0);
/// // Using default sampler for objective's domain.
/// let random_obj = (variable.get_sampler_obj())(&variable.get_domain_obj(), &mut rng);
/// assert!(variable.get_domain_obj().is_in(&random_obj));
///
/// // Default Optimizer part (Clone of domain_obj)
/// assert_eq!(variable.get_domain_opt().lower(),0.0);
/// assert_eq!(variable.get_domain_opt().upper(),10.0);
/// // Using default sampler for optimizer's domain.
/// let random_opt = (variable.get_sampler_opt())(&variable.get_domain_opt(),&mut rng);
/// assert!(variable.get_domain_opt().is_in(&random_opt));
/// ```
///
/// - Create a [`Var`] with the [`Domain`] of the [`Objective`] function
///   and the [`Domain`] of the [`Optimizer`] algorithm
///
/// ```
/// use tantale::core::domain::{Real, Domain, DomainBounded};
/// use tantale::core::variable::Var;
/// use tantale::core::var;
///
/// let domobj = Real::new(80.0, 100.0);
/// let domopt = Real::new(0.0, 1.0);
/// let variable = var!("a" ; domobj ; domopt);
///
/// assert_eq!(variable.get_name(),("a", None));
///
/// let mut rng = rand::rng();
///
/// // Objective part (Defined above)
/// assert_eq!(variable.get_domain_obj().lower(),80.0);
/// assert_eq!(variable.get_domain_obj().upper(),100.0);
/// // Using default sampler for objective's domain.
/// let random_obj = (variable.get_sampler_obj())(&variable.get_domain_obj(),&mut rng);
/// assert!(variable.get_domain_obj().is_in(&random_obj));
///
/// // Optimizer part (Defined above)
/// assert_eq!(variable.get_domain_opt().lower(),0.0);
/// assert_eq!(variable.get_domain_opt().upper(),1.0);
/// // Using default sampler for optimizer's domain.
/// let random_opt = (variable.get_sampler_opt())(&variable.get_domain_opt(),&mut rng);
/// assert!(variable.get_domain_opt().is_in(&random_opt));
/// ```
///
/// - Create a [`Var`] with the [`Domain`] of the [`Objective`] function
///   and a sampler.
///
/// ```
/// use tantale::core::domain::{Real, Domain, DomainBounded};
/// use tantale::core::variable::Var;
/// use tantale::core::domain::sampler::uniform_real;
/// use tantale::core::var;
///
/// let domobj = Real::new(80.0, 100.0);
/// let variable = var!("a" ; domobj => uniform_real);
///
/// assert_eq!(variable.get_name(),("a", None));
///
/// let mut rng = rand::rng();
///
/// // Objective part (Defined above)
/// assert_eq!(variable.get_domain_obj().lower(),80.0);
/// assert_eq!(variable.get_domain_obj().upper(),100.0);
/// // Using given sampler for objective's domain.
/// let random_obj = (variable.get_sampler_obj())(&variable.get_domain_obj(),&mut rng);
/// assert!(variable.get_domain_obj().is_in(&random_obj));
///
/// // Default Optimizer part (Clone of domain_obj)
/// assert_eq!(variable.get_domain_opt().lower(),80.0);
/// assert_eq!(variable.get_domain_opt().upper(),100.0);
/// // Using default sampler for optimizer's domain.
/// let random_opt = (variable.get_sampler_opt())(&variable.get_domain_opt(),&mut rng);
/// assert!(variable.get_domain_opt().is_in(&random_opt));
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
/// let variable = var!("a" ; domobj);
///
/// let domobj = Int::new(0, 100);
/// let variable = var!("a" ; domobj => uniform_int);
///
/// let domobj = Int::new(0, 100);
/// let variable = var!("a" ; domobj => uniform_int ; => uniform_int);
///
/// let domobj = Int::new(0, 100);
/// let variable = var!("a" ; domobj ; => uniform_int);
///
/// let domobj = Int::new(0, 100);
/// let domopt = Real::new(0.0, 1.0);
/// let variable = var!("a" ; domobj ; domopt);
///
/// let domobj = Int::new(0, 100);
/// let domopt = Real::new(0.0, 1.0);
/// let variable = var!("a" ; domobj => uniform_int ; domopt => uniform_real);
///
/// let domobj = Int::new(0, 100);
/// let domopt = Real::new(0.0, 1.0);
/// let variable = var!("a" ; domobj => uniform_int ; domopt);
///
/// let domobj = Int::new(0, 100);
/// let domopt = Real::new(0.0, 1.0);
/// let variable = var!("a" ; domobj ; domopt => uniform_real);
/// ```
#[macro_export]
macro_rules! var {
    // Defining both objective and optimizer domains
    // Defining both samplers
    ($name:literal ; $domobj:expr => $sampobj:expr ; $domopt:expr => $sampopt:expr) => {{
        $crate::variable::Var::new_double(
            ($name, None),
            std::sync::Arc::new($domobj),
            std::sync::Arc::new($domopt),
            Some($sampobj),
            Some($sampopt),
        )
    }};
    // Solely defining objective sampler
    ($name:literal ; $domobj:expr => $sampobj:expr ; $domopt:expr) => {{
        $crate::variable::Var::new_double(
            ($name, None),
            std::sync::Arc::new($domobj),
            std::sync::Arc::new($domopt),
            Some($sampobj),
            None,
        )
    }};
    // Solely defining optimizer sampler
    ($name:literal ; $domobj:expr ; $domopt:expr => $sampopt:expr) => {{
        $crate::variable::Var::new_double(
            ($name, None),
            std::sync::Arc::new($domobj),
            std::sync::Arc::new($domopt),
            None,
            Some($sampopt),
        )
    }};
    // No sampler
    ($name:literal ; $domobj:expr ; $domopt:expr) => {{
        $crate::variable::Var::new_double(
            ($name, None),
            std::sync::Arc::new($domobj),
            std::sync::Arc::new($domopt),
            None,
            None,
        )
    }};

    // Solely defining objective domain
    // Defining both samplers
    ($name:literal ; $domobj:expr => $sampobj:expr ; => $sampopt:expr) => {{
        $crate::variable::Var::new_single(
            ($name, None),
            std::sync::Arc::new($domobj),
            Some($sampobj),
            Some($sampopt),
        )
    }};
    // Solely defining objective sampler
    ($name:literal ; $domobj:expr => $sampobj:expr) => {{
        $crate::variable::Var::new_single(
            ($name, None),
            std::sync::Arc::new($domobj),
            Some($sampobj),
            None,
        )
    }};
    // Solely defining optimizer sampler
    ($name:literal ;  $domobj:expr ; => $sampopt:expr) => {{
        $crate::variable::Var::new_single(
            ($name, None),
            std::sync::Arc::new($domobj),
            None,
            Some($sampopt),
        )
    }};
    // No sampler
    ($name:literal ; $domobj:expr) => {{
        $crate::variable::Var::new_single(($name, None), std::sync::Arc::new($domobj), None, None)
    }};
}
