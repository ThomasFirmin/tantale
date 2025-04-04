/// # Variable
/// This crate describes what a variable is.
/// Most of the domains implements the [`Domain`] type trait [`TypeDom`]. It gives the type of a point within this domain.
/// Domains are use in [`crate::core::variable::Variable`] to define the type of the variable, its `TypeObjective` and `TypeOptimizer`, repectively the input type of that variable within the [`crate::core::objective:Objective`] function, and the input type of the [`crate::core::optimizer::Optimizer`].
///
use crate::core::domain::Domain;
use rand::prelude::ThreadRng;

pub trait DomainObjective {
    type TypeObj;
    fn sample_obj(&self, rng: &mut ThreadRng) -> Self::TypeObj;
}
pub trait DomainOptimizer {
    type TypeOpt;
    fn sample_opt(&self, rng: &mut ThreadRng) -> Self::TypeOpt;
}
pub struct Variable<'a, T: Domain, U: Domain> {
    pub name: &'a str,
    pub domain_obj: T,
    pub domain_opt: U,
    sampler_obj: fn(&T, &mut ThreadRng) -> T::TypeDom,
    sampler_opt: fn(&U, &mut ThreadRng) -> U::TypeDom,
}

impl<T: Domain, U: Domain> DomainObjective for Variable<'_, T, U> {
    type TypeObj = T::TypeDom;
    fn sample_obj(&self, rng: &mut ThreadRng) -> Self::TypeObj {
        (self.sampler_obj)(&self.domain_obj, rng)
    }
}
impl<T: Domain, U: Domain> DomainOptimizer for Variable<'_, T, U> {
    type TypeOpt = U::TypeDom;
    fn sample_opt(&self, rng: &mut ThreadRng) -> Self::TypeOpt {
        (self.sampler_opt)(&self.domain_opt, rng)
    }
}

impl<'a, T: Domain, U: Domain> Variable<'a, T, U> {
    pub fn new(
        name: &'a str,
        domobj: T,
        domopt: U,
        sampobj: Option<fn(&T, &mut ThreadRng) -> T::TypeDom>,
        sampopt: Option<fn(&U, &mut ThreadRng) -> U::TypeDom>,
    ) -> Variable<'a, T, U> {
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

    pub fn sample_obj(&self, rng: &mut ThreadRng) -> <Self as DomainObjective>::TypeObj {
        (self.sampler_obj)(&self.domain_obj, rng)
    }

    pub fn sample_opt(&self, rng: &mut ThreadRng) -> <Self as DomainOptimizer>::TypeOpt {
        (self.sampler_opt)(&self.domain_opt, rng)
    }
}
