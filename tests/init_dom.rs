pub use tantale::core::domain::Domain;
pub use tantale::core::domain::bool::Bool;
pub use tantale::core::domain::cat::Cat;
pub use tantale::core::domain::bounded::{Int,Nat,Real,Bounded,BoundedBounds};
pub use tantale::core::domain::unit::Unit;
pub use tantale::core::domain::base::{BaseDom,BaseTypeDom};
pub use tantale::core::domain::onto::Onto;

const ACTIVATION : [&str; 3] = ["relu", "tanh", "sigmoid"];

pub type DefaultBasereal<'a> = BaseDom<'a, 0, f64>;
pub type DefaultBaseTreal<'a> = BaseTypeDom<'a, 0, f64>;
pub type DefaultBasenat<'a> = BaseDom<'a, 0, u64>;
pub type DefaultBaseTnat<'a> = BaseTypeDom<'a, 0, u64>;
pub type DefaultBaseint<'a> = BaseDom<'a, 0, i64>;
pub type DefaultBaseTint<'a> = BaseTypeDom<'a, 0, i64>;
pub type DefaultBasebool<'a> = BaseDom<'a, 0, u8>;
pub type DefaultBaseTbool<'a> = BaseTypeDom<'a, 0, u8>;
pub type DefaultBasecat<'a> = BaseDom<'a, 3, u8>;
pub type DefaultBaseTcat<'a> = BaseTypeDom<'a, 3, u8>;
pub type DefaultBaseunit<'a> = BaseDom<'a, 0, u8>;
pub type DefaultBaseTunit<'a> = BaseTypeDom<'a, 0, u8>;

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

pub fn get_domain_cat_2<'a>() -> Cat<'a, 3> {
    return Cat::new(&ACTIVATION);
}

pub fn get_domain_unit_2() -> Unit{
    return Unit::new();
}




pub fn get_domain_base_real<'a>(domain:Real,input:f64) -> (DefaultBasereal<'a>,DefaultBaseTreal<'a>) {
    return (BaseDom::Bounded(domain),BaseTypeDom::Bounded(input));
}

pub fn get_domain_base_nat<'a>(domain:Nat,input:u64) -> (DefaultBasenat<'a>,DefaultBaseTnat<'a>) {
    return (BaseDom::Bounded(domain),BaseTypeDom::Bounded(input));
}

pub fn get_domain_base_int<'a>(domain:Int,input:i64) -> (DefaultBaseint<'a>,DefaultBaseTint<'a>) {
    return (BaseDom::Bounded(domain),BaseTypeDom::Bounded(input));
}

pub fn get_domain_base_bool<'a>(domain:Bool,input:bool) -> (DefaultBasebool<'a>,DefaultBaseTbool<'a>) {
    return (BaseDom::Bool(domain),BaseTypeDom::Bool(input));
}

pub fn get_domain_base_cat<'a>(domain:Cat<'a,3>,input:&'a str) -> (DefaultBasecat<'a>,DefaultBaseTcat<'a>) {
    return (BaseDom::Cat(domain),BaseTypeDom::Cat(input));
}

pub fn get_domain_base_unit<'a>(domain:Unit,input:f64) -> (DefaultBaseunit<'a>,DefaultBaseTunit<'a>) {
    return (BaseDom::Unit(domain),BaseTypeDom::Unit(input));
}