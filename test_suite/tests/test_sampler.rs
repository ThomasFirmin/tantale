mod check_sampler {
    use tantale::core::domain::sampler::*;
    use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real};

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
        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
        let cat_1 = Cat::new(&ACTIVATION);
        assert!(
            cat_1.is_in(&uniform_cat(&cat_1, &mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
}

mod check_sampler_base {
    use tantale::core::domain::sampler::*;
    use tantale::core::domain::{BaseDom, Bool, Cat, Domain, Int, Nat, Real, Unit};
    use tantale::core::mixed_sampler;

    #[test]
    fn sampler_real_base() {
        let mut rng = rand::rng();
        let real_1: BaseDom = Real::new(0.0, 10.0).into();
        mixed_sampler!(let sampler : Real = uniform_real);

        assert!(
            real_1.is_in(&sampler(&real_1, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Real"
        );
    }
    #[test]
    fn sampler_nat_base() {
        let mut rng = rand::rng();
        let nat_1: BaseDom = Nat::new(0, 10).into();

        mixed_sampler!(let sampler : Nat = uniform_nat);
        assert!(
            nat_1.is_in(&sampler(&nat_1, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Nat"
        );
    }
    #[test]
    fn sampler_int_base() {
        let mut rng = rand::rng();
        let int_1: BaseDom = Int::new(0, 10).into();

        mixed_sampler!(let sampler : Int = uniform_int);
        assert!(
            int_1.is_in(&sampler(&int_1, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Int"
        );
    }
    #[test]
    fn sampler_bool_base() {
        let mut rng = rand::rng();
        let bool_1: BaseDom = Bool::new().into();
        mixed_sampler!(let sampler : Bool = uniform_bool);

        assert!(
            bool_1.is_in(&sampler(&bool_1, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Bool"
        );
    }
    #[test]
    fn sampler_cat_base() {
        let mut rng = rand::rng();
        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
        let cat_1: BaseDom = Cat::new(&ACTIVATION).into();
        mixed_sampler!(let sampler : Cat = uniform_cat);
        assert!(
            cat_1.is_in(&sampler(&cat_1, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Cat"
        );
    }
    #[test]
    fn sampler_unit_base() {
        let mut rng = rand::rng();
        let unit_1: BaseDom = Unit::new().into();
        mixed_sampler!(let sampler : Unit = uniform_unit);

        assert!(
            unit_1.is_in(&sampler(&unit_1, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Unit"
        );
    }
    #[test]
    fn sampler_base_many() {
        let mut rng = rand::rng();
        let real_1: BaseDom = Real::new(0.0, 10.0).into();
        let real_2: BaseDom = Real::new(100.0, 1000.0).into();
        mixed_sampler!(let sampler_real : Real = uniform_real);

        let nat_1: BaseDom = Nat::new(0, 10).into();
        let nat_2: BaseDom = Nat::new(100, 1000).into();
        mixed_sampler!(let sampler_nat : Nat = uniform_nat);

        let int_1: BaseDom = Int::new(0, 10).into();
        let int_2: BaseDom = Int::new(100, 1000).into();
        mixed_sampler!(let sampler_int : Int = uniform_int);

        let bool_1: BaseDom = Bool::new().into();
        let bool_2: BaseDom = Bool::new().into();
        mixed_sampler!(let sampler_bool : Bool = uniform_bool);

        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
        let cat_1: BaseDom = Cat::new(&ACTIVATION).into();
        let cat_2: BaseDom = Cat::new(&ACTIVATION).into();
        mixed_sampler!(let sampler_cat : Cat = uniform_cat);

        let unit_1: BaseDom = Unit::new().into();
        let unit_2: BaseDom = Unit::new().into();
        mixed_sampler!(let sampler_unit : Unit = uniform_unit);

        assert!(
            real_1.is_in(&sampler_real(&real_1, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Real"
        );
        assert!(
            real_2.is_in(&sampler_real(&real_2, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Real"
        );

        assert!(
            nat_1.is_in(&sampler_nat(&nat_1, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Nat"
        );
        assert!(
            nat_2.is_in(&sampler_nat(&nat_2, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Nat"
        );

        assert!(
            int_1.is_in(&sampler_int(&int_1, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Int"
        );
        assert!(
            int_2.is_in(&sampler_int(&int_2, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Int"
        );

        assert!(
            bool_1.is_in(&sampler_bool(&bool_1, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Bool"
        );
        assert!(
            bool_2.is_in(&sampler_bool(&bool_2, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Bool"
        );

        assert!(
            cat_1.is_in(&sampler_cat(&cat_1, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Cat"
        );
        assert!(
            cat_2.is_in(&sampler_cat(&cat_2, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Cat"
        );

        assert!(
            unit_1.is_in(&sampler_unit(&unit_1, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Unit"
        );
        assert!(
            unit_2.is_in(&sampler_unit(&unit_2, &mut rng)),
            "Error while sampling with the default sampler of BaseDom::Unit"
        );
    }
}
