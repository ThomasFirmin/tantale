use tantale_macros::sp;

#[test]
fn searchspace_test(){
    
    use tantale_core::domain::{Real,Int,Nat,Bool,Cat,Unit};
    use tantale_core::domain::sampler::{uniform_real,uniform_nat};

    static ACTIVATION: [&str;3] = ["relu","tanh","sigmoid"];

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,1)       ;
        b | Nat(0,1)                      | => uniform_nat ;
        c | Bool()                        |                ;
    )

    
}