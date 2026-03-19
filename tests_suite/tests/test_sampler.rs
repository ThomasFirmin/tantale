mod check_sampler {
    use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real};
    use tantale::core::{GridInt, GridNat, GridReal, sampler::*};

    #[test]
    fn sampler_real() {
        let mut rng = rand::rng();
        let real_1 = Real::new(0.0, 10.0, Uniform);
        assert!(
            real_1.is_in(&real_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_nat() {
        let mut rng = rand::rng();
        let nat_1 = Nat::new(0, 10, Uniform);
        assert!(
            nat_1.is_in(&nat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_int() {
        let mut rng = rand::rng();
        let int_1 = Int::new(0, 10, Uniform);
        assert!(
            int_1.is_in(&int_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Int"
        );
    }
    #[test]
    fn sampler_bool() {
        let mut rng = rand::rng();
        let bool_1 = Bool::new(Bernoulli(0.5));
        assert!(
            bool_1.is_in(&bool_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_cat() {
        let mut rng = rand::rng();
        let cat_1 = Cat::new(["relu", "tanh", "sigmoid"], Uniform);
        assert!(
            cat_1.is_in(&cat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }

    #[test]
    fn sampler_greal() {
        let mut rng = rand::rng();
        let greal_1 = GridReal::new([-2.0, -1.0, 0.0, 1.0, 2.0], Uniform);
        assert!(
            greal_1.is_in(&greal_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_gnat() {
        let mut rng = rand::rng();
        let gnat_1 = GridNat::new([0_u64, 10, 20, 30], Uniform);
        assert!(
            gnat_1.is_in(&gnat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_gint() {
        let mut rng = rand::rng();
        let gint_1 = GridInt::new([-50_i64, -25, 0, 25, 50], Uniform);
        assert!(
            gint_1.is_in(&gint_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
}

mod check_sampler_base {
    use tantale::core::domain::{Bool, Cat, Domain, Int, Mixed, Nat, Real, Unit};
    use tantale::core::{GridInt, GridNat, GridReal, sampler::*};

    #[test]
    fn sampler_real_base() {
        let mut rng = rand::rng();
        let mixed: Mixed = Real::new(0.0, 10.0, Uniform).into();

        assert!(
            mixed.is_in(&mixed.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Real"
        );
    }
    #[test]
    fn sampler_nat_base() {
        let mut rng = rand::rng();
        let mixed: Mixed = Nat::new(0, 10, Uniform).into();

        assert!(
            mixed.is_in(&mixed.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Nat"
        );
    }
    #[test]
    fn sampler_int_base() {
        let mut rng = rand::rng();
        let mixed: Mixed = Int::new(0, 10, Uniform).into();

        assert!(
            mixed.is_in(&mixed.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Int"
        );
    }
    #[test]
    fn sampler_bool_base() {
        let mut rng = rand::rng();
        let mixed: Mixed = Bool::new(Bernoulli(0.5)).into();

        assert!(
            mixed.is_in(&mixed.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Bool"
        );
    }
    #[test]
    fn sampler_cat_base() {
        let mut rng = rand::rng();
        let mixed: Mixed = Cat::new(["relu", "tanh", "sigmoid"], Uniform).into();

        assert!(
            mixed.is_in(&mixed.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Cat"
        );
    }
    #[test]
    fn sampler_unit_base() {
        let mut rng = rand::rng();
        let mixed: Mixed = Unit::new(Uniform).into();

        assert!(
            mixed.is_in(&mixed.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Unit"
        );
    }

    #[test]
    fn sampler_greal_base() {
        let mut rng = rand::rng();
        let mixed: Mixed = GridReal::new([-2.0, -1.0, 0.0, 1.0, 2.0], Uniform).into();

        assert!(
            mixed.is_in(&mixed.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::GridReal"
        );
    }

    #[test]
    fn sampler_gnat_base() {
        let mut rng = rand::rng();
        let mixed: Mixed = GridNat::new([0_u64, 10, 20, 30], Uniform).into();

        assert!(
            mixed.is_in(&mixed.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::GridNat"
        );
    }

    #[test]
    fn sampler_gint_base() {
        let mut rng = rand::rng();
        let mixed: Mixed = GridInt::new([-50_i64, -25, 0, 25, 50], Uniform).into();

        assert!(
            mixed.is_in(&mixed.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::GridInt"
        );
    }

    #[test]
    fn sampler_base_many() {
        let mut rng = rand::rng();
        let mixed_real_1: Mixed = Real::new(0.0, 10.0, Uniform).into();
        let mixed_real_2: Mixed = Real::new(100.0, 1000.0, Uniform).into();

        let mixed_nat_1: Mixed = Nat::new(0, 10, Uniform).into();
        let mixed_nat_2: Mixed = Nat::new(100, 1000, Uniform).into();

        let mixed_int_1: Mixed = Int::new(0, 10, Uniform).into();
        let mixed_int_2: Mixed = Int::new(100, 1000, Uniform).into();

        let mixed_bool_1: Mixed = Bool::new(Bernoulli(0.5)).into();
        let mixed_bool_2: Mixed = Bool::new(Bernoulli(0.5)).into();

        let mixed_cat_1: Mixed = Cat::new(["relu", "tanh", "sigmoid"], Uniform).into();
        let mixed_cat_2: Mixed = Cat::new(["relu", "tanh", "sigmoid"], Uniform).into();

        let mixed_unit_1: Mixed = Unit::new(Uniform).into();
        let mixed_unit_2: Mixed = Unit::new(Uniform).into();

        let mixed_greal_1: Mixed = GridReal::new([-2.0, -1.0, 0.0, 1.0, 2.0], Uniform).into();
        let mixed_greal_2: Mixed = GridReal::new([-30.0, -20.0, 10.0, 20.0], Uniform).into();

        let mixed_gnat_1: Mixed = GridNat::new([0_u64, 10, 20, 30], Uniform).into();
        let mixed_gnat_2: Mixed = GridNat::new([100_u64, 200, 300, 400], Uniform).into();

        let mixed_gint_1: Mixed = GridInt::new([-50_i64, -25, 0, 25, 50], Uniform).into();
        let mixed_gint_2: Mixed = GridInt::new([-500_i64, -250, 250, 500], Uniform).into();

        assert!(
            mixed_real_1.is_in(&mixed_real_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Real"
        );
        assert!(
            mixed_real_2.is_in(&mixed_real_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Real"
        );

        assert!(
            mixed_nat_1.is_in(&mixed_nat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Nat"
        );
        assert!(
            mixed_nat_2.is_in(&mixed_nat_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Nat"
        );

        assert!(
            mixed_int_1.is_in(&mixed_int_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Int"
        );
        assert!(
            mixed_int_2.is_in(&mixed_int_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Int"
        );

        assert!(
            mixed_bool_1.is_in(&mixed_bool_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Bool"
        );
        assert!(
            mixed_bool_2.is_in(&mixed_bool_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Bool"
        );

        assert!(
            mixed_cat_1.is_in(&mixed_cat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Cat"
        );
        assert!(
            mixed_cat_2.is_in(&mixed_cat_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Cat"
        );

        assert!(
            mixed_unit_1.is_in(&mixed_unit_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Unit"
        );
        assert!(
            mixed_unit_2.is_in(&mixed_unit_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Unit"
        );

        assert!(
            mixed_greal_1.is_in(&mixed_greal_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::GridReal"
        );
        assert!(
            mixed_greal_2.is_in(&mixed_greal_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::GridReal"
        );

        assert!(
            mixed_gnat_1.is_in(&mixed_gnat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::GridNat"
        );
        assert!(
            mixed_gnat_2.is_in(&mixed_gnat_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::GridNat"
        );

        assert!(
            mixed_gint_1.is_in(&mixed_gint_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::GridInt"
        );
        assert!(
            mixed_gint_2.is_in(&mixed_gint_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::GridInt"
        );
    }
}
