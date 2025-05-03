
pub mod searchspace{
    use tantale_macros::sp;
    use tantale_core::domain::{Real,Int,Nat,Bool};
    use tantale_core::domain::sampler::{uniform_real,uniform_nat};
    sp!(
        a | Real(0.0,1.0) => uniform_real |                ;
        b | Nat(0,1)                      | => uniform_nat ;
        c | Bool()                        |                ;
    );
}
#[test]
fn searchspace_test(){
    searchspace::get_searchpace();
}