pub use tantale::core::domain::base::{BaseDom, BaseTypeDom};
pub use tantale::core::domain::bool::Bool;
pub use tantale::core::domain::bounded::{Bounded, BoundedBounds, Int, Nat, Real};
pub use tantale::core::domain::cat::Cat;
pub use tantale::core::domain::onto::Onto;
pub use tantale::core::domain::unit::Unit;
pub use tantale::core::domain::Domain;
use tantale_core::sampler::{Bernoulli, Uniform};

pub fn get_domain_real() -> Real {
    Real::new(0.0, 10.0, Uniform)
}

pub fn get_domain_nat() -> Nat {
    Nat::new(1, 11, Uniform)
}

pub fn get_domain_int() -> Int {
    Int::new(0, 10, Uniform)
}

pub fn get_domain_bool() -> Bool {
    Bool::new(Bernoulli(0.5))
}

pub fn get_domain_cat() -> Cat {
    Cat::new(["relu", "tanh", "sigmoid"], Uniform)
}

pub fn get_domain_unit() -> Unit {
    Unit::new(Uniform)
}

pub fn get_domain_real_2() -> Real {
    Real::new(80.0, 100.0, Uniform)
}

pub fn get_domain_nat_2() -> Nat {
    Nat::new(80, 100, Uniform)
}

pub fn get_domain_int_2() -> Int {
    Int::new(80, 100, Uniform)
}

pub fn get_domain_bool_2() -> Bool {
    Bool::new(Bernoulli(0.5))
}

pub fn get_domain_cat_2() -> Cat {
    Cat::new(["relu", "tanh", "sigmoid"], Uniform)
}

pub fn get_domain_unit_2() -> Unit {
    Unit::new(Uniform)
}

pub fn get_domain_base_real(domain: Real, input: f64) -> (BaseDom, BaseTypeDom) {
    (BaseDom::Real(domain), BaseTypeDom::Real(input))
}

pub fn get_domain_base_nat(domain: Nat, input: u64) -> (BaseDom, BaseTypeDom) {
    (BaseDom::Nat(domain), BaseTypeDom::Nat(input))
}

pub fn get_domain_base_int(domain: Int, input: i64) -> (BaseDom, BaseTypeDom) {
    (BaseDom::Int(domain), BaseTypeDom::Int(input))
}

pub fn get_domain_base_bool(domain: Bool, input: bool) -> (BaseDom, BaseTypeDom) {
    (BaseDom::Bool(domain), BaseTypeDom::Bool(input))
}

pub fn get_domain_base_cat(domain: Cat, input: String) -> (BaseDom, BaseTypeDom) {
    (BaseDom::Cat(domain), BaseTypeDom::Cat(input))
}

pub fn get_domain_base_unit(domain: Unit, input: f64) -> (BaseDom, BaseTypeDom) {
    (BaseDom::Unit(domain), BaseTypeDom::Unit(input))
}
