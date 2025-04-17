pub use tantale::core::domain::{Bool, Cat, Int, Nat, Real,Unit};
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

pub fn get_domain_cat<'a>() -> Cat<'a, 3> {
    return Cat::new(&ACTIVATION);
}

pub fn get_domain_unit() -> Unit<f64> {
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

pub fn get_domain_cat_2<'a>() -> Cat<'a, 3> {
    return Cat::new(&ACTIVATION);
}

pub fn get_domain_unit_2() -> Unit<f64> {
    return Unit::new();
}