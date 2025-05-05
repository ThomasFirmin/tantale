
pub mod searchspace{
    use tantale_macros::sp;
    use tantale_core::domain::{Real,Int,Nat,Bool,Cat};
    use tantale_core::domain::sampler::{uniform_real,uniform_nat,uniform_int};
    
    static ACTIVATION : [&str;3] = ["relu","tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) => uniform_real      |  Int(0,100) => uniform_int ;
        b | Real(0.0,1.0)                      |  Nat(0,100) => uniform_nat ;
        c | Real(0.0,1.0)                      |  Cat(&ACTIVATION)  ;
        d | Real(0.0,1.0)                      |  Bool()          ;
    );

}
#[test]
fn searchspace_test(){
    let var = searchspace::get_searchpace();
    
    let mut rng = rand::rng();

    for v in &var{
        println!("{} - {}", v.sample_obj(&mut rng), v.sample_opt(&mut rng))
    }
    println!("\n OBJ to OPT \n");
    for v in &var{
        let sample = v.sample_obj(&mut rng);
        println!("{} - {}", sample, v.onto_opt(&sample).unwrap())
    }
    println!("\n OPT to OBJ \n");
    for v in &var{
        let sample = v.sample_opt(&mut rng);
        println!("{} - {}", v.onto_obj(&sample).unwrap(), sample)
    }
}