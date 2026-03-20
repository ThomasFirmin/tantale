mod check_bounds {
    use tantale::core::sampler::{Bernoulli, Uniform};
    use tantale::core::{Bool, Cat, Domain, GridInt, GridNat, GridReal, Int, Nat, Real, Unit};
    #[test]
    fn create_real() {
        let real_1 = Real::new(0.0, 10.0, Uniform);
        assert_eq!(
            real_1.bounds.start(),
            &0.0,
            "Issue with lower bound of Real."
        );
        assert_eq!(
            real_1.bounds.end(),
            &10.0,
            "Issue with upper bound of Real."
        );

        assert!(
            real_1.is_in(&0.0),
            "Issue with is_in for lower bound of Real."
        );
        assert!(
            real_1.is_in(&5.0),
            "Issue with is_in for mid value from Real."
        );
        assert!(
            real_1.is_in(&10.0),
            "Issue with is_in for upper bound of Real."
        );
        assert!(
            !real_1.is_in(&-1.0),
            "Issue with is_in with value < lower bound of Real."
        );
        assert!(
            !real_1.is_in(&11.0),
            "Issue with is_in with value > upper bound of Real."
        );
    }
    #[test]
    #[should_panic]
    fn fail_real_bounds() {
        Real::new(10.1, 0.1, Uniform);
    }
    #[test]
    fn create_nat() {
        let nat_1 = Nat::new(1, 10, Uniform);
        assert_eq!(nat_1.bounds.start(), &1, "Issue with lower bound of Nat.");
        assert_eq!(nat_1.bounds.end(), &10, "Issue with upper bound of Nat.");

        assert!(nat_1.is_in(&1), "Issue with is_in for lower bound of Nat.");
        assert!(nat_1.is_in(&5), "Issue with is_in for mid value from Nat.");
        assert!(nat_1.is_in(&10), "Issue with is_in for upper bound of Nat.");
        assert!(
            !nat_1.is_in(&0),
            "Issue with is_in with value < lower bound of Nat."
        );
        assert!(
            !nat_1.is_in(&11),
            "Issue with is_in with value > upper bound of Nat."
        );
    }
    #[test]
    #[should_panic]
    fn fail_nat_bounds() {
        Nat::new(10, 0, Uniform);
    }
    #[test]
    fn create_int() {
        let int_1 = Int::new(0, 10, Uniform);
        assert_eq!(int_1.bounds.start(), &0, "Issue with lower bound of Int.");
        assert_eq!(int_1.bounds.end(), &10, "Issue with upper bound of Int.");

        assert!(int_1.is_in(&0), "Issue with is_in for lower bound of Int.");
        assert!(int_1.is_in(&5), "Issue with is_in for mid value from Int.");
        assert!(int_1.is_in(&10), "Issue with is_in for upper bound of Int.");
        assert!(
            !int_1.is_in(&-1),
            "Issue with is_in with value < lower bound of Int."
        );
        assert!(
            !int_1.is_in(&11),
            "Issue with is_in with value > upper bound of Int."
        );
    }
    #[test]
    #[should_panic]
    fn fail_int_bounds() {
        Int::new(10, 0, Uniform);
    }
    #[test]
    fn bool() {
        let bool_1 = Bool::new(Bernoulli(0.5));

        assert!(bool_1.is_in(&true), "Issue with is_in for `true` of Bool.");
        assert!(
            bool_1.is_in(&false),
            "Issue with is_in for `false` of Bool."
        );
    }
    #[test]
    fn create_cat() {
        let check = vec![
            String::from("relu"),
            String::from("tanh"),
            String::from("sigmoid"),
        ];
        let cat_1 = Cat::new(["relu", "tanh", "sigmoid"], Uniform);
        let values = cat_1.values.to_vec();
        assert_eq!(&values, &check, "Issue with content of Cat.");

        assert!(
            cat_1.is_in(&String::from("relu")),
            "Issue with is_in for the first element of Cat."
        );
        assert!(
            cat_1.is_in(&String::from("tanh")),
            "Issue with is_in for the second element of Cat."
        );
        assert!(
            cat_1.is_in(&String::from("sigmoid")),
            "Issue with is_in for the third element of Cat."
        );
        assert!(
            !cat_1.is_in(&String::from("a")),
            "Issue with is_in with value not in values of Cat."
        );
    }
    #[test]
    fn create_unit64() {
        let unit: Unit = Unit::new(Uniform);
        assert_eq!(unit.bounds.start(), &0.0, "Issue with lower bound of Unit.");
        assert_eq!(unit.bounds.end(), &1.0, "Issue with upper bound of Unit.");

        assert!(
            unit.is_in(&0.0),
            "Issue with is_in for lower bound of Unit."
        );
        assert!(
            unit.is_in(&0.5),
            "Issue with is_in for mid value from Unit."
        );
        assert!(
            unit.is_in(&1.0),
            "Issue with is_in for upper bound of Unit."
        );
        assert!(
            !unit.is_in(&-1.0),
            "Issue with is_in with value < lower bound of Unit."
        );
        assert!(
            !unit.is_in(&11.0),
            "Issue with is_in with value > upper bound of Unit."
        );
    }
    #[test]
    fn create_gridreal() {
        let check = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let greal_1 = GridReal::new([-2.0, -1.0, 0.0, 1.0, 2.0], Uniform);
        let values = greal_1.values.to_vec();
        assert_eq!(&values, &check, "Issue with content of GridReal.");

        assert!(
            greal_1.is_in(&-2.0),
            "Issue with is_in for the first element of GridReal."
        );
        assert!(
            greal_1.is_in(&-1.0),
            "Issue with is_in for the second element of GridReal."
        );
        assert!(
            greal_1.is_in(&0.0),
            "Issue with is_in for the third element of GridReal."
        );
        assert!(
            greal_1.is_in(&1.0),
            "Issue with is_in for the fourth element of GridReal."
        );
        assert!(
            greal_1.is_in(&2.0),
            "Issue with is_in for the fifth element of GridReal."
        );
        assert!(
            !greal_1.is_in(&3.0),
            "Issue with is_in with value not in values of GridReal."
        );
    }
    #[test]
    fn create_gridint() {
        let check = vec![-2, -1, 0, 1, 2];
        let greal_1 = GridInt::new([-2_i64, -1, 0, 1, 2], Uniform);
        let values = greal_1.values.to_vec();
        assert_eq!(&values, &check, "Issue with content of GridInt.");

        assert!(
            greal_1.is_in(&-2),
            "Issue with is_in for the first element of GridInt."
        );
        assert!(
            greal_1.is_in(&-1),
            "Issue with is_in for the second element of GridInt."
        );
        assert!(
            greal_1.is_in(&0),
            "Issue with is_in for the third element of GridInt."
        );
        assert!(
            greal_1.is_in(&1),
            "Issue with is_in for the fourth element of GridInt."
        );
        assert!(
            greal_1.is_in(&2),
            "Issue with is_in for the fifth element of GridInt."
        );
        assert!(
            !greal_1.is_in(&3),
            "Issue with is_in with value not in values of GridInt."
        );
    }
    #[test]
    fn create_gridnat() {
        let check = vec![0, 1, 2];
        let greal_1 = GridNat::new([0_u64, 1, 2], Uniform);
        let values = greal_1.values.to_vec();
        assert_eq!(&values, &check, "Issue with content of GridNat.");

        assert!(
            greal_1.is_in(&0),
            "Issue with is_in for the first element of GridNat."
        );
        assert!(
            greal_1.is_in(&1),
            "Issue with is_in for the second element of GridNat."
        );
        assert!(
            greal_1.is_in(&2),
            "Issue with is_in for the third element of GridNat."
        );
        assert!(
            !greal_1.is_in(&3),
            "Issue with is_in with value not in values of GridNat."
        );
    }
}

mod check_mid {
    use tantale::core::sampler::Uniform;
    use tantale::core::{Int, Nat, Real, Unit};

    #[test]
    fn mid_real() {
        let real_1 = Real::new(0.0, 10.0, Uniform);
        assert_eq!(real_1.mid, 5.0, "Error for mid of DomainBounded Real.");
    }
    #[test]
    fn mid_nat_even() {
        let nat_1 = Nat::new(0, 10, Uniform);
        assert_eq!(nat_1.mid, 5, "Error for mid of DomainBounded Nat.");
    }
    #[test]
    fn mid_int_even() {
        let int_1 = Int::new(0, 10, Uniform);
        assert_eq!(int_1.mid, 5, "Error for mid of DomainBounded Int.");
    }
    #[test]
    fn mid_nat_odd() {
        let nat_1 = Nat::new(0, 11, Uniform);
        assert_eq!(nat_1.mid, 5, "Error for odd mid of DomainBounded Nat.");
    }
    #[test]
    fn mid_int_odd() {
        let int_1 = Int::new(0, 11, Uniform);
        assert_eq!(int_1.mid, 5, "Error for odd mid of DomainBounded Int.");
    }
    #[test]
    fn mid_unit64() {
        let unit: Unit = Unit::new(Uniform);
        assert_eq!(unit.mid, 0.5, "Error for mid of DomainBounded Unit<64>.");
    }
    #[test]
    fn mid_unit32() {
        let unit: Unit = Unit::new(Uniform);
        assert_eq!(unit.mid, 0.5, "Error for mid of DomainBounded Unit<f32>.");
    }
}

mod check_width {
    use tantale::core::sampler::Uniform;
    use tantale::core::{Int, Nat, Real};

    #[test]
    fn width_real_zero() {
        let real_1 = Real::new(0.0, 10.0, Uniform);
        assert_eq!(real_1.width, 10.0, "Error for range of DomainBounded Real.");
    }
    #[test]
    fn width_nat_zero() {
        let nat_1 = Nat::new(0, 10, Uniform);
        assert_eq!(nat_1.width, 10, "Error for range of DomainBounded Nat.");
    }
    #[test]
    fn width_int_zero() {
        let int_1 = Int::new(0, 10, Uniform);
        assert_eq!(int_1.width, 10, "Error for range of DomainBounded Int.");
    }
    #[test]
    fn width_real_nzero() {
        let real_1 = Real::new(1.0, 11.0, Uniform);
        assert_eq!(real_1.width, 10.0, "Error for range of DomainBounded Real.");
    }
    #[test]
    fn width_nat_nzero() {
        let nat_1 = Nat::new(1, 11, Uniform);
        assert_eq!(nat_1.width, 10, "Error for odd range of DomainBounded Nat.");
    }
    #[test]
    fn width_int_nzero() {
        let int_1 = Int::new(1, 11, Uniform);
        assert_eq!(int_1.width, 10, "Error for odd range of DomainBounded Int.");
    }
}

mod check_default_sampler {
    use tantale::core::sampler::{Bernoulli, Uniform};
    use tantale::core::{Bool, Cat, Domain, GridInt, GridNat, GridReal, Int, Nat, Real, Unit};
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
    fn sampler_unit() {
        let mut rng = rand::rng();
        let unit_1: Unit = Unit::new(Uniform);
        assert!(
            unit_1.is_in(&unit_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Unit"
        );
    }

    #[test]
    fn sampler_gridreal() {
        let mut rng = rand::rng();
        let gridreal_1 = GridReal::new([-2.0, -1.0, 0.0, 1.0, 2.0], Uniform);
        assert!(
            gridreal_1.is_in(&gridreal_1.sample(&mut rng)),
            "Error while sampling with the default sampler of GridReal"
        );
    }
    #[test]
    fn sampler_gridnat() {
        let mut rng = rand::rng();
        let gridnat_1 = GridNat::new([0_u64, 1, 2, 3, 4], Uniform);
        assert!(
            gridnat_1.is_in(&gridnat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of GridNat"
        );
    }
    #[test]
    fn sampler_gridint() {
        let mut rng = rand::rng();
        let gridint_1 = GridInt::new([-2, -1, 0, 1, 2], Uniform);
        assert!(
            gridint_1.is_in(&gridint_1.sample(&mut rng)),
            "Error while sampling with the default sampler of GridInt"
        );
    }
}

mod check_default_sampler_mixed {
    use tantale::core::sampler::{Bernoulli, Uniform};
    use tantale::core::{
        Bool, Cat, Domain, GridInt, GridNat, GridReal, Int, Mixed, Nat, Real, Unit,
    };

    #[test]
    fn sampler_real() {
        let mut rng = rand::rng();
        let real_1 = Mixed::Real(Real::new(0.0, 10.0, Uniform));
        assert!(
            real_1.is_in(&real_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_nat() {
        let mut rng = rand::rng();
        let nat_1 = Mixed::Nat(Nat::new(0, 10, Uniform));
        assert!(
            nat_1.is_in(&nat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Nat"
        );
    }
    #[test]
    fn sampler_int() {
        let mut rng = rand::rng();
        let int_1 = Mixed::Int(Int::new(0, 10, Uniform));
        assert!(
            int_1.is_in(&int_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Int"
        );
    }
    #[test]
    fn sampler_bool() {
        let mut rng = rand::rng();
        let bool_1 = Mixed::Bool(Bool::new(Bernoulli(0.5)));
        assert!(
            bool_1.is_in(&bool_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Bool"
        );
    }
    #[test]
    fn sampler_cat() {
        let mut rng = rand::rng();
        let cat_1 = Mixed::Cat(Cat::new(["relu", "tanh", "sigmoid"], Uniform));
        assert!(
            cat_1.is_in(&cat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Cat"
        );
    }
    #[test]
    fn sampler_unit() {
        let mut rng = rand::rng();
        let unit_1 = Mixed::Unit(Unit::new(Uniform));
        assert!(
            unit_1.is_in(&unit_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Unit"
        );
    }
    #[test]
    fn sampler_gridreal() {
        let mut rng = rand::rng();
        let gridreal_1 = Mixed::GridReal(GridReal::new([-2.0, -1.0, 0.0, 1.0, 2.0], Uniform));
        assert!(
            gridreal_1.is_in(&gridreal_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Mixed::GridReal"
        );
    }
    #[test]
    fn sampler_gridnat() {
        let mut rng = rand::rng();
        let gridnat_1 = Mixed::GridNat(GridNat::new([0_u64, 1, 2, 3, 4], Uniform));
        assert!(
            gridnat_1.is_in(&gridnat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Mixed::GridNat"
        );
    }
    #[test]
    fn sampler_gridint() {
        let mut rng = rand::rng();
        let gridint_1 = Mixed::GridInt(GridInt::new([-2, -1, 0, 1, 2], Uniform));
        assert!(
            gridint_1.is_in(&gridint_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Mixed::GridInt"
        );
    }
}
