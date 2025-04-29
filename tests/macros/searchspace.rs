use tantale_macros::sp;

#[test]
fn searchspace_test(){
    
    use tantale_core::domain::{Real,Int,Nat,Bool,Cat,Unit};
    use tantale_core::domain::sampler::{uniform_real,uniform_nat};

    let v1 = Real::new(0.0,1.0);
    let v2 = Int::new(0,1);
    let v3 = Nat::new(0,1);
    let v4 = Bool::new();
    static ACTIVATION: [&str;3] = ["relu","tanh","sigmoid"];
    let v5 = Cat::new(&ACTIVATION);
    let v6 = Unit::new();

    sp!(
        a | v1 => uniform_real | v2 ;
        b | v3                 | => uniform_nat;
        c | v4                 | ;
    )
}