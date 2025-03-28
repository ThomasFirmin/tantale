mod check_bounds {
    use tantale::core::domain::{self, Domain, NumericallyBounded};
    #[test]
    fn test_real() {
        let real_1 = domain::Real::new(0.0, 10.0).expect("Error while creating Real");
        assert_eq!(real_1.lower(), 0.0, "Issue with lower bound of Real.");
        assert_eq!(real_1.upper(), 10.0, "Issue with upper bound of Real.");
        assert_eq!(real_1.bounds(), (0.0, 10.0), "Issue with bounds of Real.");

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
    #[should_panic(expected = "Error while creating Real: Boundaries error, 10.1 is not < 0.1.")]
    fn test_fail_real_bounds() {
        domain::Real::new(10.1, 0.1).expect("Error while creating Real");
    }
    #[test]
    fn test_nat() {
        let nat_1 = domain::Nat::new(1, 10).expect("Error while creating Nat");
        assert_eq!(nat_1.lower(), 1, "Issue with lower bound of Nat.");
        assert_eq!(nat_1.upper(), 10, "Issue with upper bound of Nat.");
        assert_eq!(nat_1.bounds(), (1, 10), "Issue with bounds of Nat.");

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
    #[should_panic(expected = "Error while creating Nat: Boundaries error, 0 - 10 is not > 1.")]
    fn test_fail_nat_bounds() {
        domain::Nat::new(10, 0).expect("Error while creating Nat");
    }
    #[test]
    #[should_panic(expected = "Error while creating Nat: Boundaries error, 2 - 1 is not > 1.")]
    fn test_fail_nat_short_bounds() {
        domain::Nat::new(1, 2).expect("Error while creating Nat");
    }
    #[test]
    fn test_int() {
        let int_1 = domain::Int::new(0, 10).expect("Error while creating Int");
        assert_eq!(int_1.lower(), 0, "Issue with lower bound of Int.");
        assert_eq!(int_1.upper(), 10, "Issue with upper bound of Int.");
        assert_eq!(int_1.bounds(), (0, 10), "Issue with bounds of Int.");

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
    #[should_panic(expected = "Error while creating Int: Boundaries error, 0 - 10 is not > 1.")]
    fn test_fail_int_bounds() {
        domain::Int::new(10, 0).expect("Error while creating Int");
    }
    #[test]
    #[should_panic(expected = "Error while creating Int: Boundaries error, 2 - 1 is not > 1.")]
    fn test_fail_int_short_bounds() {
        domain::Int::new(1, 2).expect("Error while creating Int");
    }
    #[test]
    fn test_bool() {
        let bool_1 = domain::Bool::new().expect("Error while creating bool");
        assert_eq!(
            bool_1.values(),
            (true, false),
            "Issue with values() of Bool."
        );

        assert!(bool_1.is_in(&true), "Issue with is_in for `true` of Bool.");
        assert!(bool_1.is_in(&false), "Issue with is_in for `false` of Bool.");
    }
    #[test]
    fn test_cat() {
        let activation = ["relu", "tanh", "sigmoid"];
        let check = ["relu", "tanh", "sigmoid"];
        let cat_1 = domain::Cat::new(activation).expect("Error while creating Cat");
        assert_eq!(cat_1.values(), check, "Issue with content of Cat.");

        assert!(
            cat_1.is_in(&"relu"),
            "Issue with is_in for the first element of Cat."
        );
        assert!(
            cat_1.is_in(&"tanh"),
            "Issue with is_in for the second element of Cat."
        );
        assert!(
            cat_1.is_in(&"sigmoid"),
            "Issue with is_in for the third element of Cat."
        );
        assert!(
            !cat_1.is_in(&"a"),
            "Issue with is_in with value not in values of Cat."
        );
    }
}

mod check_mid {
    use tantale::core::domain::{Int, Nat, NumericallyBounded, Real};

    #[test]
    fn test_mid_real() {
        let real_1 = Real::new(0.0, 10.0).unwrap();
        assert_eq!(
            real_1.mid(),
            5.0,
            "Error for mid of NumericallyBounded Real."
        );
    }
    #[test]
    fn test_mid_nat_even() {
        let nat_1 = Nat::new(0, 10).unwrap();
        assert_eq!(nat_1.mid(), 5, "Error for mid of NumericallyBounded Nat.");
    }
    #[test]
    fn test_mid_int_even() {
        let int_1 = Int::new(0, 10).unwrap();
        assert_eq!(int_1.mid(), 5, "Error for mid of NumericallyBounded Int.");
    }
    #[test]
    fn test_mid_nat_odd() {
        let nat_1 = Nat::new(0, 11).unwrap();
        assert_eq!(
            nat_1.mid(),
            5,
            "Error for odd mid of NumericallyBounded Nat."
        );
    }
    #[test]
    fn test_mid_int_odd() {
        let int_1 = Int::new(0, 11).unwrap();
        assert_eq!(
            int_1.mid(),
            5,
            "Error for odd mid of NumericallyBounded Int."
        );
    }
}

mod check_domtype {
    use tantale::core::domain::{self, Domain};

    #[test]
    fn test_isf64() {
        trait IsF64 {}
        impl IsF64 for f64 {}
        fn check_if_isf64<T: Domain>() -> bool
        where
            T::TypeDom: IsF64,
        {
            true
        }
        assert!(
            check_if_isf64::<domain::Real>(),
            "Real does not have a f64 VarType"
        );
    }

    #[test]
    fn tests_usize() {
        trait IsUsize {}
        impl IsUsize for u64 {}
        impl IsUsize for u32 {}
        impl IsUsize for u16 {}
        impl IsUsize for u8 {}
        impl IsUsize for usize {}
        fn check_if_isusize<T: Domain>() -> bool
        where
            T::TypeDom: IsUsize,
        {
            true
        }
        assert!(
            check_if_isusize::<domain::Nat>(),
            "Nat does not have a usize VarType"
        );
    }

    #[test]
    fn tests_isize() {
        trait IsIsize {}
        impl IsIsize for i64 {}
        impl IsIsize for i32 {}
        impl IsIsize for i16 {}
        impl IsIsize for i8 {}
        impl IsIsize for isize {}
        fn check_if_isisize<T: Domain>() -> bool
        where
            T::TypeDom: IsIsize,
        {
            true
        }
        assert!(
            check_if_isisize::<domain::Int>(),
            "Int does not have a isize VarType"
        );
    }

    #[test]
    fn test_isbool() {
        trait IsBool {}
        impl IsBool for bool {}
        fn check_if_isbool<T: Domain>() -> bool
        where
            T::TypeDom: IsBool,
        {
            true
        }
        assert!(
            check_if_isbool::<domain::Bool>(),
            "Bool does not have a bool VarType"
        );
    }

    #[test]
    fn test_isstr() {
        trait IsStrRef<'a> {}
        impl IsStrRef<'_> for &'_ str {}
        fn check_if_isstr<T: Domain>()
        where
            T::TypeDom: for<'b> IsStrRef<'b>,
        {
            assert!(true, "Cat does not have a &str VarType.");
        }
        check_if_isstr::<domain::Cat<'_, 3>>();
    }
}

mod check_default_sampler {
    use rand;
    use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real};
    #[test]
    fn test_sampler_real() {
        let mut rng = rand::rng();
        let real_1 = Real::new(0.0, 10.0).expect("Error while creating Real");
        let sampler = real_1.default_sampler();
        assert!(
            real_1.is_in(&sampler(&real_1, &mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn test_sampler_nat() {
        let mut rng = rand::rng();
        let nat_1 = Nat::new(0, 10).expect("Error while creating Nat");
        let sampler = nat_1.default_sampler();
        assert!(
            nat_1.is_in(&sampler(&nat_1, &mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn test_sampler_int() {
        let mut rng = rand::rng();
        let int_1 = Int::new(0, 10).expect("Error while creating Int");
        let sampler = int_1.default_sampler();
        assert!(
            int_1.is_in(&sampler(&int_1, &mut rng)),
            "Error while sampling with the default sampler of Int"
        );
    }
    #[test]
    fn test_sampler_bool() {
        let mut rng = rand::rng();
        let bool_1 = Bool::new().expect("Error while creating Bool");
        let sampler = bool_1.default_sampler();
        assert!(
            bool_1.is_in(&sampler(&bool_1, &mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn test_sampler_cat() {
        let mut rng = rand::rng();
        let activation = ["relu", "tanh", "sigmoid"];
        let cat_1 = Cat::new(activation).expect("Error while creating Cat");
        let sampler = cat_1.default_sampler();
        assert!(
            cat_1.is_in(&sampler(&cat_1, &mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
}

mod check_into_real {
    #[test]
    fn test_real_into_real() {
        use tantale::core::domain::Real;
        use tantale::core::convertible::Convertible;

        let real_1 = Real::new(0.0, 10.0).expect("Error while creating Real 1");
        let real_2 = Real::new(80.0, 100.0).expect("Error while creating Real 2");

        let point = 5.0;

        let mapped = real_1.to_real(&point, &real_2).expect("Error in mapping middle from Real to Real");
        assert_eq!(mapped,90.0,"Mapping middle of Real to Real does not match")
    }
}

mod check_into_nat {}

mod check_into_int {}

mod check_into_bool {}

mod check_into_cat {}
