pub use tantale::core::domain::base::{BaseDom, BaseTypeDom};
pub use tantale::core::domain::bool::Bool;
pub use tantale::core::domain::bounded::{Bounded, BoundedBounds, Int, Nat, Real};
pub use tantale::core::domain::cat::Cat;
pub use tantale::core::domain::onto::Onto;
pub use tantale::core::domain::unit::Unit;
pub use tantale::core::domain::Domain;

static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

pub fn get_domain_real() -> Real {
    return Real::new(0.0, 10.0);
}

pub fn get_domain_nat() -> Nat {
    return Nat::new(1, 11);
}

pub fn get_domain_int() -> Int {
    return Int::new(0, 10);
}

pub fn get_domain_bool() -> Bool {
    return Bool::new();
}

pub fn get_domain_cat() -> Cat {
    return Cat::new(&ACTIVATION);
}

pub fn get_domain_unit() -> Unit {
    return Unit::new();
}

pub fn get_domain_real_2() -> Real {
    return Real::new(80.0, 100.0);
}

pub fn get_domain_nat_2() -> Nat {
    return Nat::new(80, 100);
}

pub fn get_domain_int_2() -> Int {
    return Int::new(80, 100);
}

pub fn get_domain_bool_2() -> Bool {
    return Bool::new();
}

pub fn get_domain_cat_2() -> Cat {
    return Cat::new(&ACTIVATION);
}

pub fn get_domain_unit_2() -> Unit {
    return Unit::new();
}

pub fn get_domain_base_real(domain: Real, input: f64) -> (BaseDom, BaseTypeDom) {
    return (BaseDom::Real(domain), BaseTypeDom::Real(input));
}

pub fn get_domain_base_nat(domain: Nat, input: u64) -> (BaseDom, BaseTypeDom) {
    return (BaseDom::Nat(domain), BaseTypeDom::Nat(input));
}

pub fn get_domain_base_int(domain: Int, input: i64) -> (BaseDom, BaseTypeDom) {
    return (BaseDom::Int(domain), BaseTypeDom::Int(input));
}

pub fn get_domain_base_bool(domain: Bool, input: bool) -> (BaseDom, BaseTypeDom) {
    return (BaseDom::Bool(domain), BaseTypeDom::Bool(input));
}

pub fn get_domain_base_cat(domain: Cat, input: &'static str) -> (BaseDom, BaseTypeDom) {
    return (BaseDom::Cat(domain), BaseTypeDom::Cat(input));
}

pub fn get_domain_base_unit(domain: Unit, input: f64) -> (BaseDom, BaseTypeDom) {
    return (BaseDom::Unit(domain), BaseTypeDom::Unit(input));
}
