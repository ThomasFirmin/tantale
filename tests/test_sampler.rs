mod check_sampler {
    use rand;
    use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real};
    use tantale::core::domain::sampler::*;

    #[test]
    fn sampler_real() {
        let mut rng = rand::rng();
        let real_1 = Real::new(0.0, 10.0);
        assert!(
            real_1.is_in(&uniform_real(&real_1, &mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_nat() {
        let mut rng = rand::rng();
        let nat_1 = Nat::new(0, 10);
        assert!(
            nat_1.is_in(&uniform_nat(&nat_1, &mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_int() {
        let mut rng = rand::rng();
        let int_1 = Int::new(0, 10);
        assert!(
            int_1.is_in(&uniform_int(&int_1, &mut rng)),
            "Error while sampling with the default sampler of Int"
        );
    }
    #[test]
    fn sampler_bool() {
        let mut rng = rand::rng();
        let bool_1 = Bool::new();
        assert!(
            bool_1.is_in(&uniform_bool(&bool_1, &mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_cat() {
        let mut rng = rand::rng();
        let activation = ["relu", "tanh", "sigmoid"];
        let cat_1 = Cat::new(&activation);
        assert!(
            cat_1.is_in(&uniform_cat(&cat_1, &mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
}