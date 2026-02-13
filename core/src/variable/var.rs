//! Variable linking mechanism between [`Domain`]s.
//!
//! A [`Var`] ties together two related [`Domain`]s in the context of optimization:
//! - The **Objective Domain** (`Obj`): The space in which the objective function operates
//! - The **Optimizer Domain** (`Opt`): The space in which the optimizer searches
//!
//! # Core Concept
//!
//! In AutoML and optimization, it's often useful to work with different representations
//! of the same conceptual variable. For example:
//! - An optimizer might work best in a normalized `[0, 1]` space ([`Unit`](crate::domain::Unit))
//! - But the objective function requires real values in `[0, 100]` ([`Real`](crate::domain::Real))
//!
//! A [`Var`] handles the bidirectional mapping between these domains using [`Onto`](crate::Onto) traits,
//! allowing seamless conversion while maintaining type safety.
//!
//! # Two Usage Patterns
//!
//! ## 1. Linked Domains (`Var<Obj, Opt>`)
//!
//! When `Obj` and `Opt` are different domains that can be mapped between each other:
//!
//! ```
//! use tantale::core::{
//!     domain::{Real, Unit, Uniform},
//!     variable::Var,
//! };
//!
//! // Optimizer works in [0, 1], objective function expects [0, 100]
//! let var = Var::new("learning_rate", Real::new(0.0, 100.0, Uniform), Unit::new(Uniform));
//!
//! let mut rng = rand::rng();
//! let opt_value = var.sample_opt(&mut rng); // Sample from Unit [0, 1]
//! let obj_value = var.onto_obj(&opt_value).unwrap(); // Map to Real [0, 100]
//!
//! assert!(opt_value >= 0.0 && opt_value <= 1.0);
//! assert!(obj_value >= 0.0 && obj_value <= 100.0);
//! ```
//!
//! ## 2. Single Domain (`Var<Obj, NoDomain>`)
//!
//! When both optimizer and objective work in the same domain:
//!
//! ```
//! use tantale::core::{
//!     domain::{Real, NoDomain, Uniform},
//!     variable::Var,
//! };
//!
//! // Both optimizer and objective use the same Real domain
//! let var = Var::new("temperature", Real::new(0.0, 100.0, Uniform), NoDomain);
//!
//! let mut rng = rand::rng();
//! let value = var.sample_obj(&mut rng);
//! // No mapping needed - value is used directly
//! ```
//!
//! # Variable Naming
//!
//! Each [`Var`] has a name represented as a tuple `(&'static str, Option<usize>)`:
//! - The first element is the base name
//! - The second element is an optional numeric suffix (used for replication)
//!
//! # Replication
//!
//! Variables can be replicated to create arrays of similar variables, useful for
//! representing vectors or matrices in optimization problems:
//!
//! ```
//! use tantale::core::{
//!     domain::{Real, Unit, Uniform},
//!     variable::Var,
//! };
//!
//! let var = Var::new("weight", Real::new(-1.0, 1.0, Uniform), Unit::new(Uniform));
//! let weights = var.replicate(10); // Creates weight0, weight1, ..., weight9
//!
//! assert_eq!(weights.len(), 10);
//! assert_eq!(weights[0].name, ("weight", Some(0)));
//! assert_eq!(weights[5].name, ("weight", Some(5)));
//! ```
//!
//! # Integration with Tantale
//!
//! While [`Var`] can be created manually, it's typically generated automatically by
//! the [`objective!`](crate::objective!) and [`sp!`](crate::searchspace!) procedural macros,
//! which handle complex scenarios like [`Mixed`](crate::domain::Mixed) domains.

use crate::{
    domain::{
        Domain, NoDomain, PreDomain,
        onto::{LinkObj, LinkOpt, LinkTyObj, LinkTyOpt, Linked, OntoDom},
    },
    errors::OntoError,
    recorder::csv::{CSVLeftRight, CSVWritable},
};

use rand::prelude::Rng;
use std::sync::Arc;

/// A variable that links an Objective [`Domain`] and an Optimizer [`Domain`].
///
/// [`Var`] provides bidirectional mapping between two domains using [`OntoDom`] traits,
/// enabling optimizers to work in their preferred space while objective functions
/// operate in theirs.
///
/// # Type Parameters
///
/// * `Obj` - The objective function's [`Domain`]
/// * `Opt` - The optimizer's [`Domain`] (can be [`NoDomain`] if same as `Obj`)
///
/// # Fields
///
/// * `name` - A tuple `(&'static str, Option<usize>)` for identification
/// * `domain_obj` - The objective domain wrapped in [`Arc`] for efficient cloning
/// * `domain_opt` - The optimizer domain wrapped in [`Arc`] for efficient cloning
///
/// # See Also
///
/// * [`OntoDom`] - Trait for domain mapping
/// * [`Domain`] - Base trait for all domains
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

/// Implementation of [`Linked`] for variables with two distinct domains.
///
/// This establishes the type relationships between the objective and optimizer domains,
/// allowing the type system to properly track conversions.
impl<Obj: OntoDom<Opt>, Opt: OntoDom<Obj>> Linked for Var<Obj, Opt> {
    type Obj = Obj;
    type Opt = Opt;
}

/// Implementation of [`Linked`] for variables with a single domain.
///
/// When using [`NoDomain`], both objective and optimizer use the same domain,
/// so no mapping is needed.
impl<Obj: Domain> Linked for Var<Obj, NoDomain> {
    type Obj = Obj;
    type Opt = Obj;
}

/// Methods for [`Var`] with distinct objective and optimizer domains.
impl<Obj: OntoDom<Opt>, Opt: OntoDom<Obj>> Var<Obj, Opt> {
    /// Creates a new [`Var`] linking two different domains.
    ///
    /// # Parameters
    ///
    /// * `name` - Static string identifier for the variable
    /// * `domain_obj` - The domain for the objective function
    /// * `domain_opt` - The domain for the optimizer
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// // Create a variable mapping between Unit [0,1] and Real [0,100]
    /// let var = Var::new("alpha", Real::new(0.0, 100.0, Uniform), Unit::new(Uniform));
    /// assert_eq!(var.name, ("alpha", None));
    /// ```
    pub fn new(name: &'static str, domain_obj: Obj, domain_opt: Opt) -> Var<Obj, Opt> {
        Var {
            name: (name, None),
            domain_obj: Arc::new(domain_obj),
            domain_opt: Arc::new(domain_opt),
        }
    }

    /// Maps a value from the optimizer domain to the objective domain.
    ///
    /// This is the key transformation that allows optimizers to work in their preferred
    /// representation while providing values in the format expected by the objective function.
    ///
    /// # Parameters
    ///
    /// * `item` - A reference to a value in the optimizer domain
    ///
    /// # Returns
    ///
    /// * `Ok(value)` - The mapped value in the objective domain
    /// * `Err(OntoError)` - If the mapping fails (e.g., value out of bounds)
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("param", Real::new(0.0, 100.0, Uniform), Unit::new(Uniform));
    ///
    /// // Optimizer provides a value in [0, 1]
    /// let opt_value = 0.75;
    ///
    /// // Map to objective domain [0, 100]
    /// let obj_value = var.onto_obj(&opt_value).unwrap();
    /// assert_eq!(obj_value, 75.0);
    /// ```
    pub fn onto_obj(&self, item: &LinkTyOpt<Self>) -> Result<LinkTyObj<Self>, OntoError> {
        self.domain_opt.onto(item, &self.domain_obj)
    }

    /// Maps a value from the objective domain to the optimizer domain.
    ///
    /// This is useful for initialization or when converting results back to the
    /// optimizer's representation. Less commonly used than [`onto_obj`](Self::onto_obj)
    /// but important for bidirectional workflows.
    ///
    /// # Parameters
    ///
    /// * `item` - A reference to a value in the objective domain
    ///
    /// # Returns
    ///
    /// * `Ok(value)` - The mapped value in the optimizer domain
    /// * `Err(OntoError)` - If the mapping fails
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("param", Real::new(0.0, 100.0, Uniform), Unit::new(Uniform));
    ///
    /// // Objective domain value [0, 100]
    /// let obj_value = 25.0;
    ///
    /// // Map to optimizer domain [0, 1]
    /// let opt_value = var.onto_opt(&obj_value).unwrap();
    /// assert_eq!(opt_value, 0.25);
    /// ```
    pub fn onto_opt(&self, item: &LinkTyObj<Self>) -> Result<LinkTyOpt<Self>, OntoError> {
        self.domain_obj.onto(item, &self.domain_opt)
    }

    /// Samples a random value from the objective domain.
    ///
    /// Generates a valuefrom the objective domain's bounds, by using internal [`Domain`] sampler.
    /// Useful for initialization or testing with values in the objective's native space.
    ///
    /// # Parameters
    ///
    /// * `rng` - A random number generator implementing [`Rng`](rand::Rng)
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("temperature", Real::new(0.0, 100.0, Uniform), Unit::new(Uniform));
    ///
    /// let mut rng = rand::rng();
    /// let value = var.sample_obj(&mut rng);
    ///
    /// // Value should be within objective domain bounds
    /// assert!(value >= 0.0 && value <= 100.0);
    /// ```
    pub fn sample_obj<R: Rng>(&self, rng: &mut R) -> LinkTyObj<Self> {
        LinkObj::<Self>::sample(&self.domain_obj, rng)
    }
    /// Samples a random value from the optimizer domain.
    ///
    /// Generates a value within the optimizer domain's bounds, by using internal [`Domain`] sampler.
    /// This is the primary sampling method used during optimization to generate
    /// candidate solutions.
    ///
    /// # Parameters
    ///
    /// * `rng` - A random number generator implementing [`Rng`](rand::Rng)
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("alpha", Real::new(0.0, 100.0, Uniform), Unit::new(Uniform));
    ///
    /// let mut rng = rand::rng();
    /// let opt_value = var.sample_opt(&mut rng);
    ///
    /// // Value in optimizer domain [0, 1]
    /// assert!(opt_value >= 0.0 && opt_value <= 1.0);
    ///
    /// // Can be mapped to objective domain
    /// let obj_value = var.onto_obj(&opt_value).unwrap();
    /// assert!(obj_value >= 0.0 && obj_value <= 100.0);
    /// ```
    pub fn sample_opt<R: Rng>(&self, rng: &mut R) -> LinkTyOpt<Self> {
        LinkOpt::<Self>::sample(&self.domain_opt, rng)
    }
    /// Checks if a value is within the objective domain's bounds.
    ///
    /// Validates whether a given value satisfies the constraints of the objective domain.
    /// Useful for boundary checking and constraint validation.
    ///
    /// # Parameters
    ///
    /// * `item` - A reference to a value to check
    ///
    /// # Returns
    ///
    /// * `true` if the value is within bounds
    /// * `false` otherwise
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("param", Real::new(0.0, 100.0, Uniform), Unit::new(Uniform));
    ///
    /// assert!(var.is_in_obj(&50.0));
    /// assert!(var.is_in_obj(&0.0));
    /// assert!(var.is_in_obj(&100.0));
    /// assert!(!var.is_in_obj(&-10.0));
    /// assert!(!var.is_in_obj(&150.0));
    /// ```
    pub fn is_in_obj(&self, item: &LinkTyObj<Self>) -> bool {
        LinkObj::<Self>::is_in(&self.domain_obj, item)
    }
    /// Checks if a value is within the optimizer domain's bounds.
    ///
    /// Validates whether a given value satisfies the constraints of the optimizer domain.
    /// Used to ensure optimizer-generated values are valid before mapping.
    ///
    /// # Parameters
    ///
    /// * `item` - A reference to a value to check
    ///
    /// # Returns
    ///
    /// * `true` if the value is within bounds
    /// * `false` otherwise
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("param", Real::new(0.0, 100.0, Uniform), Unit::new(Uniform));
    ///
    /// // Unit domain is [0, 1]
    /// assert!(var.is_in_opt(&0.5));
    /// assert!(var.is_in_opt(&0.0));
    /// assert!(var.is_in_opt(&1.0));
    /// assert!(!var.is_in_opt(&1.5));
    /// assert!(!var.is_in_opt(&-0.1));
    /// ```
    pub fn is_in_opt(&self, item: &LinkTyOpt<Self>) -> bool {
        LinkOpt::<Self>::is_in(&self.domain_opt, item)
    }
    /// Replicates the variable a specified number of times.
    ///
    /// Creates a vector of variables sharing the same domains but with indexed names.
    /// Domains are shared via [`Arc`] cloning, making replication efficient. Each
    /// replicated variable gets a numeric suffix in its name.
    ///
    /// This is useful for representing vectors, matrices, or collections of similar
    /// parameters in optimization problems.
    ///
    /// # Parameters
    ///
    /// * `repeats` - Number of replicas to create
    ///
    /// # Returns
    ///
    /// A vector of `repeats` variables with indexed names
    ///
    /// # Notes
    ///
    /// This method consumes `self` since the original name structure changes.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, Unit, Uniform},
    ///     variable::Var,
    /// };
    /// use std::sync::Arc;
    ///
    /// let var = Var::new("weight", Real::new(-1.0, 1.0, Uniform), Unit::new(Uniform));
    /// let weights = var.replicate(5);
    ///
    /// assert_eq!(weights.len(), 5);
    /// assert_eq!(weights[0].name, ("weight", Some(0)));
    /// assert_eq!(weights[1].name, ("weight", Some(1)));
    /// assert_eq!(weights[4].name, ("weight", Some(4)));
    ///
    /// // All variables share the same domain (Arc pointers are equal)
    /// assert!(Arc::ptr_eq(&weights[0].domain_obj, &weights[1].domain_obj));
    /// ```
    pub fn replicate(self, repeats: usize) -> Vec<Self> {
        let mut vec = Vec::with_capacity(repeats);
        for i in 0..repeats {
            let var = Var {
                name: (self.name.0, Some(i)),
                domain_obj: self.domain_obj.clone(),
                domain_opt: self.domain_opt.clone(),
            };
            vec.push(var);
        }
        vec
    }
}

/// Methods for [`Var`] with a single shared domain ([`NoDomain`]).
///
/// When both optimizer and objective work in the same domain, this implementation
/// provides simplified semantics where mapping operations are identity functions.
/// This is useful when no transformation is needed between optimizer and objective spaces.
impl<Obj: Domain> Var<Obj, NoDomain> {
    /// Creates a new [`Var`] with a single domain shared by both objective and optimizer.
    ///
    /// # Parameters
    ///
    /// * `name` - Static string identifier for the variable
    /// * `domain_obj` - The domain used by both objective and optimizer
    /// * `domain_opt` - Must be [`NoDomain`]
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, NoDomain, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// // Both optimizer and objective use Real domain directly
    /// let var = Var::new("param", Real::new(0.0, 100.0, Uniform), NoDomain);
    /// assert_eq!(var.name, ("param", None));
    /// ```
    pub fn new(name: &'static str, domain_obj: Obj, domain_opt: NoDomain) -> Var<Obj, NoDomain> {
        Var {
            name: (name, None),
            domain_obj: Arc::new(domain_obj),
            domain_opt: Arc::new(domain_opt),
        }
    }

    /// Maps a value from optimizer domain to objective domain (identity operation).
    ///
    /// Since both domains are the same, this simply clones the input value.
    /// It is preferable to compute with [`Lone`](crate::solution::Lone) [`SolutionShape`](crate::solution::SolutionShape)
    /// when using a single domain, but this method is provided for completeness.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, NoDomain},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("x", Real::new(0.0, 10.0, Uniform), NoDomain);
    /// let value = 5.0;
    /// assert_eq!(var.onto_obj(&value).unwrap(), 5.0);
    /// ```
    pub fn onto_obj(&self, item: &LinkTyOpt<Self>) -> Result<LinkTyObj<Self>, OntoError> {
        Ok(item.clone())
    }

    /// Maps a value from objective domain to optimizer domain (identity operation).
    ///
    /// Since both domains are the same, this simply clones the input value.
    /// It is preferable to compute with [`Lone`](crate::solution::Lone) [`SolutionShape`](crate::solution::SolutionShape)
    /// when using a single domain, but this method is provided for completeness.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, NoDomain, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("x", Real::new(0.0, 10.0, Uniform), NoDomain);
    /// let value = 7.5;
    /// assert_eq!(var.onto_opt(&value).unwrap(), 7.5);
    /// ```
    pub fn onto_opt(&self, item: &LinkTyObj<Self>) -> Result<LinkTyOpt<Self>, OntoError> {
        Ok(item.clone())
    }

    /// Samples a random value from the objective domain.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, NoDomain, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("x", Real::new(0.0, 10.0, Uniform), NoDomain);
    /// let mut rng = rand::rng();
    /// let value = var.sample_obj(&mut rng);
    /// assert!(value >= 0.0 && value <= 10.0);
    /// ```
    pub fn sample_obj<R: Rng>(&self, rng: &mut R) -> LinkTyObj<Self> {
        LinkObj::<Self>::sample(&self.domain_obj, rng)
    }
    /// Samples a random value from the optimizer domain (same as objective).
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, NoDomain, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("x", Real::new(0.0, 10.0, Uniform), NoDomain);
    /// let mut rng = rand::rng();
    /// let value = var.sample_opt(&mut rng);
    /// assert!(value >= 0.0 && value <= 10.0);
    /// ```
    pub fn sample_opt<R: Rng>(&self, rng: &mut R) -> LinkTyOpt<Self> {
        LinkOpt::<Self>::sample(&self.domain_obj, rng)
    }
    /// Checks if a value is within the objective domain's bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, NoDomain, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("x", Real::new(0.0, 10.0, Uniform), NoDomain);
    /// assert!(var.is_in_obj(&5.0));
    /// assert!(!var.is_in_obj(&15.0));
    /// ```
    pub fn is_in_obj(&self, item: &LinkTyObj<Self>) -> bool {
        LinkObj::<Self>::is_in(&self.domain_obj, item)
    }
    /// Checks if a value is within the optimizer domain's bounds (same as objective).
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, NoDomain, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("x", Real::new(0.0, 10.0, Uniform), NoDomain);
    /// assert!(var.is_in_opt(&5.0));
    /// assert!(!var.is_in_opt(&-5.0));
    /// ```
    pub fn is_in_opt(&self, item: &LinkTyOpt<Self>) -> bool {
        LinkOpt::<Self>::is_in(&self.domain_obj, item)
    }
    /// Replicates the variable a specified number of times.
    ///
    /// Creates a vector of variables sharing the same domain with indexed names.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{
    ///     domain::{Real, NoDomain, Uniform},
    ///     variable::Var,
    /// };
    ///
    /// let var = Var::new("param", Real::new(0.0, 1.0, Uniform), NoDomain);
    /// let params = var.replicate(3);
    ///
    /// assert_eq!(params.len(), 3);
    /// assert_eq!(params[0].name, ("param", Some(0)));
    /// assert_eq!(params[2].name, ("param", Some(2)));
    /// ```
    pub fn replicate(self, repeats: usize) -> Vec<Self> {
        let mut vec = Vec::with_capacity(repeats);
        for i in 0..repeats {
            let var = Var {
                name: (self.name.0, Some(i)),
                domain_obj: self.domain_obj.clone(),
                domain_opt: self.domain_opt.clone(),
            };
            vec.push(var);
        }
        vec
    }
}

/// Implementation of [`CSVLeftRight`] for [`Var`] with distinct domains.
///
/// Enables CSV serialization of variables with separate objective and optimizer domains,
/// supporting recording and checkpointing workflows in Tantale.
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

/// Implementation of [`CSVLeftRight`] for [`Var`] with a single domain ([`NoDomain`]).
///
/// Enables CSV serialization of variables where objective and optimizer share the same domain.
impl<Obj> CSVLeftRight<Self, LinkTyObj<Self>, LinkTyOpt<Self>> for Var<Obj, NoDomain>
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
