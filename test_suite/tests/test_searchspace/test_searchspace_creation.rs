mod searchspace {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat, uniform_real};
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0)                   |                               ;
        b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
        c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
        d | Bool()                          | Real(0.0,1.0)                 ;
    );
}

#[test]
fn create_mixed_searchspace() {
    let sp = searchspace::get_searchspace();
    let var = &sp.variables;

    for v in var {
        let mut rng = rand::rng();
        let sample_obj = v.sample_obj(&mut rng);
        let converted_obj = v.onto_opt(&sample_obj).unwrap();

        println!("OBJ {} => OPT {}", sample_obj, converted_obj);

        let sample_opt = v.sample_opt(&mut rng);
        let converted_opt = v.onto_obj(&sample_opt).unwrap();

        println!("OPT {} => OBJ {}\n", sample_opt, converted_opt);
    }
}
