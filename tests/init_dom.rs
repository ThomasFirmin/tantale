pub use tantale::core::domain::Domain;
pub use tantale::core::domain::bool::Bool;
pub use tantale::core::domain::cat::Cat;
pub use tantale::core::domain::bounded::{Int,Nat,Real,Bounded,BoundedBounds};
pub use tantale::core::domain::unit::Unit;
pub use tantale::core::domain::base::{BaseDom,BaseTypeDom};
pub use tantale::core::domain::onto::Onto;

const ACTIVATION : [&str; 3] = ["relu", "tanh", "sigmoid"];

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

pub fn get_domain_cat<'a>() -> Cat<'a> {
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

pub fn get_domain_cat_2<'a>() -> Cat<'a> {
    return Cat::new(&ACTIVATION);
}

pub fn get_domain_unit_2() -> Unit{
    return Unit::new();
}




pub fn get_domain_base_real<'a>(domain:Real,input:f64) -> (BaseDom<'a>,BaseTypeDom<'a>) {
    return (BaseDom::Real(domain),BaseTypeDom::Real(input));
}

pub fn get_domain_base_nat<'a>(domain:Nat,input:u64) -> (BaseDom<'a>,BaseTypeDom<'a>) {
    return (BaseDom::Nat(domain),BaseTypeDom::Nat(input));
}

pub fn get_domain_base_int<'a>(domain:Int,input:i64) -> (BaseDom<'a>,BaseTypeDom<'a>) {
    return (BaseDom::Int(domain),BaseTypeDom::Int(input));
}

pub fn get_domain_base_bool<'a>(domain:Bool,input:bool) -> (BaseDom<'a>,BaseTypeDom<'a>) {
    return (BaseDom::Bool(domain),BaseTypeDom::Bool(input));
}

pub fn get_domain_base_cat<'a>(domain:Cat<'a>,input:&'a str) -> (BaseDom<'a>,BaseTypeDom<'a>) {
    return (BaseDom::Cat(domain),BaseTypeDom::Cat(input));
}

pub fn get_domain_base_unit<'a>(domain:Unit,input:f64) -> (BaseDom<'a>,BaseTypeDom<'a>) {
    return (BaseDom::Unit(domain),BaseTypeDom::Unit(input));
}