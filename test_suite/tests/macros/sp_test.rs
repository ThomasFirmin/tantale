pub mod searchspace{
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::{uniform_int, uniform_nat, uniform_cat, uniform_bool};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       => uniform_int  | ;
        b | Nat(0,100)       => uniform_nat  | ;
        c | Cat(&ACTIVATION) => uniform_cat  | ;
        d | Bool()           => uniform_bool | ;
    );
}
#[test]
fn searchspace_test(){
    searchspace::get_searchpace();
}