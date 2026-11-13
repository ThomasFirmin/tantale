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
//! See the crate-level documentation for complete, tested usage examples.
//!
//! ## 2. Single Domain (`Var<Obj, NoDomain>`)
//!
//! When both optimizer and objective work in the same domain:
//!
//! See the crate-level documentation for complete, tested usage examples.
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
//! See the crate-level documentation for complete, tested usage examples.
//!
//! # Integration with Tantale
//!
//! While [`Var`] can be created manually, it's typically generated automatically by
//! the `objective!` and `hpo!` procedural macros,
//! which handle complex scenarios like [`Mixed`](crate::domain::Mixed) domains.

use crate::{
    aliases::{LinkObj, LinkOpt, LinkTyObj, LinkTyOpt},
    domain::{
        Domain, NoDomain, PreDomain,
        onto::{Linked, OntoDom},
    },
    errors::OntoError,
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
    type TrueOpt = Opt;
}

/// Implementation of [`Linked`] for variables with a single domain.
///
/// When using [`NoDomain`], both objective and optimizer use the same domain,
/// so no mapping is needed.
impl<Obj: Domain> Linked for Var<Obj, NoDomain> {
    type Obj = Obj;
    type Opt = Obj;
    type TrueOpt = NoDomain;
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
/// See the crate-level documentation for complete, tested usage examples.
pub fn new(name: &'static str, domain_obj: Obj, domain_opt: Opt) -> Var<Obj, Opt> {
        Var {
            name: (name, None),
            domain_obj: Arc::new(domain_obj),
            domain_opt: Arc::new(domain_opt),
        }
    }

    /// Gets a reference to the objective domain.
    pub fn get_obj(&self) -> &Obj {
        self.domain_obj.as_ref()
    }

    /// Gets a reference to the optimizer domain.
    pub fn get_opt(&self) -> &Opt {
        self.domain_opt.as_ref()
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
/// See the crate-level documentation for complete, tested usage examples.
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
/// See the crate-level documentation for complete, tested usage examples.
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
    /// * `rng` - A random number generator implementing [`Rng`]
    ///
    /// # Example
    ///
/// See the crate-level documentation for complete, tested usage examples.
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
    /// * `rng` - A random number generator implementing [`Rng`]
    ///
    /// # Example
    ///
/// See the crate-level documentation for complete, tested usage examples.
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
/// See the crate-level documentation for complete, tested usage examples.
pub fn is_in_obj(&self, item: &LinkTyObj<Self>) -> bool {
        LinkObj::<Self>::contains(&self.domain_obj, item)
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
/// See the crate-level documentation for complete, tested usage examples.
pub fn is_in_opt(&self, item: &LinkTyOpt<Self>) -> bool {
        LinkOpt::<Self>::contains(&self.domain_opt, item)
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
/// See the crate-level documentation for complete, tested usage examples.
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
/// See the crate-level documentation for complete, tested usage examples.
pub fn new(name: &'static str, domain_obj: Obj, domain_opt: NoDomain) -> Var<Obj, NoDomain> {
        Var {
            name: (name, None),
            domain_obj: Arc::new(domain_obj),
            domain_opt: Arc::new(domain_opt),
        }
    }

    /// Gets a reference to the objective domain.
    pub fn get_obj(&self) -> &Obj {
        self.domain_obj.as_ref()
    }

    /// Gets a reference to the optimizer domain.
    pub fn get_opt(&self) -> &Obj {
        self.domain_obj.as_ref()
    }

    /// Maps a value from optimizer domain to objective domain (identity operation).
    ///
    /// Since both domains are the same, this simply clones the input value.
    /// It is preferable to compute with [`Lone`](crate::solution::Lone) [`SolutionShape`](crate::solution::SolutionShape)
    /// when using a single domain, but this method is provided for completeness.
    ///
    /// # Example
    ///
/// See the crate-level documentation for complete, tested usage examples.
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
/// See the crate-level documentation for complete, tested usage examples.
pub fn onto_opt(&self, item: &LinkTyObj<Self>) -> Result<LinkTyOpt<Self>, OntoError> {
        Ok(item.clone())
    }

    /// Samples a random value from the objective domain.
    ///
    /// # Example
    ///
/// See the crate-level documentation for complete, tested usage examples.
pub fn sample_obj<R: Rng>(&self, rng: &mut R) -> LinkTyObj<Self> {
        LinkObj::<Self>::sample(&self.domain_obj, rng)
    }
    /// Samples a random value from the optimizer domain (same as objective).
    ///
    /// # Example
    ///
/// See the crate-level documentation for complete, tested usage examples.
pub fn sample_opt<R: Rng>(&self, rng: &mut R) -> LinkTyOpt<Self> {
        LinkOpt::<Self>::sample(&self.domain_obj, rng)
    }
    /// Checks if a value is within the objective domain's bounds.
    ///
    /// # Example
    ///
/// See the crate-level documentation for complete, tested usage examples.
pub fn is_in_obj(&self, item: &LinkTyObj<Self>) -> bool {
        LinkObj::<Self>::contains(&self.domain_obj, item)
    }
    /// Checks if a value is within the optimizer domain's bounds (same as objective).
    ///
    /// # Example
    ///
/// See the crate-level documentation for complete, tested usage examples.
pub fn is_in_opt(&self, item: &LinkTyOpt<Self>) -> bool {
        LinkOpt::<Self>::contains(&self.domain_obj, item)
    }
    /// Replicates the variable a specified number of times.
    ///
    /// Creates a vector of variables sharing the same domain with indexed names.
    ///
    /// # Example
    ///
/// See the crate-level documentation for complete, tested usage examples.
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

/// Trait to convert a variable into a string representation of its name.
pub trait ToName {
    /// Converts the variable's name into a string.
    ///
    /// # Returns
    ///
    /// A `String` representing the variable's name, including any suffix if present.
    ///
    /// # Example
    ///
/// See the crate-level documentation for complete, tested usage examples.
fn to_name(&self) -> String;
}

impl<Obj: Domain, Opt: PreDomain> ToName for Var<Obj, Opt> {
    fn to_name(&self) -> String {
        match self.name.1 {
            Some(i) => format!("{}_{}", self.name.0, i),
            None => self.name.0.to_string(),
        }
    }
}

impl<Obj: Domain, Opt: PreDomain> ToName for &Var<Obj, Opt> {
    fn to_name(&self) -> String {
        match self.name.1 {
            Some(i) => format!("{}_{}", self.name.0, i),
            None => self.name.0.to_string(),
        }
    }
}
