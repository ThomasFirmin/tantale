use tantale_core::sampler::{Bernoulli, Uniform};
use tantale_macros::Mixed;

#[test]
fn mixed_derive() {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real, Unit};

    #[derive(Mixed, PartialEq)]
    pub enum Base {
        Real(Real),
        Int(Int),
        Nat(Nat),
        Bool(Bool),
        Cat(Cat),
        Unit(Unit),
    }

    let v1 = Real::new(0.0, 1.0, Uniform);
    let v2 = Int::new(0, 1, Uniform);
    let v3 = Nat::new(0, 1, Uniform);
    let v4 = Bool::new(Bernoulli(0.5));
    let v5 = Cat::new(["relu", "tanh", "sigmoid"], Uniform);
    let v6 = Unit::new(Uniform);

    let b1 = Base::Real(v1.clone());
    let b2 = Base::Int(v2.clone());
    let b3 = Base::Nat(v3.clone());
    let b4 = Base::Bool(v4);
    let b5 = Base::Cat(v5.clone());
    let b6 = Base::Unit(v6.clone());

    assert_eq!(
        match b1 {
            Base::Real(d) => d,
            _ => panic!("WRONG VARIANTS"),
        },
        v1
    );
    assert_eq!(
        match b2 {
            Base::Int(d) => d,
            _ => panic!("WRONG VARIANTS"),
        },
        v2
    );
    assert_eq!(
        match b3 {
            Base::Nat(d) => d,
            _ => panic!("WRONG VARIANTS"),
        },
        v3
    );
    assert_eq!(
        match b4 {
            Base::Bool(d) => d,
            _ => panic!("WRONG VARIANTS"),
        },
        v4
    );
    assert_eq!(
        match b5 {
            Base::Cat(d) => d,
            _ => panic!("WRONG VARIANTS"),
        },
        v5
    );
    assert_eq!(
        match b6 {
            Base::Unit(d) => d,
            _ => panic!("WRONG VARIANTS"),
        },
        v6
    );
}
