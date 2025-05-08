mod check_bounds {
    use tantale::core::{Bool, Cat, Domain, DomainBounded, Int, Nat, Real, Unit};
    #[test]
    fn create_real() {
        let real_1 = Real::new(0.0, 10.0);
        assert_eq!(real_1.lower(), 0.0, "Issue with lower bound of Real.");
        assert_eq!(real_1.upper(), 10.0, "Issue with upper bound of Real.");

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
        Real::new(10.1, 0.1);
    }
    #[test]
    fn create_nat() {
        let nat_1 = Nat::new(1, 10);
        assert_eq!(nat_1.lower(), 1, "Issue with lower bound of Nat.");
        assert_eq!(nat_1.upper(), 10, "Issue with upper bound of Nat.");

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
        Nat::new(10, 0);
    }
    #[test]
    fn create_int() {
        let int_1 = Int::new(0, 10);
        assert_eq!(int_1.lower(), 0, "Issue with lower bound of Int.");
        assert_eq!(int_1.upper(), 10, "Issue with upper bound of Int.");

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
        Int::new(10, 0);
    }
    #[test]
    fn bool() {
        let bool_1 = Bool::new();

        assert!(bool_1.is_in(&true), "Issue with is_in for `true` of Bool.");
        assert!(
            bool_1.is_in(&false),
            "Issue with is_in for `false` of Bool."
        );
    }
    #[test]
    fn create_cat() {
        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
        let check = ["relu", "tanh", "sigmoid"];
        let cat_1 = Cat::new(&ACTIVATION);
        assert_eq!(cat_1.values(), &check, "Issue with content of Cat.");

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
    #[test]
    fn create_unit64() {
        let unit: Unit = Unit::new();
        assert_eq!(unit.lower(), 0.0, "Issue with lower bound of Unit.");
        assert_eq!(unit.upper(), 1.0, "Issue with upper bound of Unit.");

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
}

mod check_mid {
    use tantale::core::{DomainBounded, Int, Nat, Real, Unit};

    #[test]
    fn mid_real() {
        let real_1 = Real::new(0.0, 10.0);
        assert_eq!(real_1.mid(), 5.0, "Error for mid of DomainBounded Real.");
    }
    #[test]
    fn mid_nat_even() {
        let nat_1 = Nat::new(0, 10);
        assert_eq!(nat_1.mid(), 5, "Error for mid of DomainBounded Nat.");
    }
    #[test]
    fn mid_int_even() {
        let int_1 = Int::new(0, 10);
        assert_eq!(int_1.mid(), 5, "Error for mid of DomainBounded Int.");
    }
    #[test]
    fn mid_nat_odd() {
        let nat_1 = Nat::new(0, 11);
        assert_eq!(nat_1.mid(), 5, "Error for odd mid of DomainBounded Nat.");
    }
    #[test]
    fn mid_int_odd() {
        let int_1 = Int::new(0, 11);
        assert_eq!(int_1.mid(), 5, "Error for odd mid of DomainBounded Int.");
    }
    #[test]
    fn mid_unit64() {
        let unit: Unit = Unit::new();
        assert_eq!(unit.mid(), 0.5, "Error for mid of DomainBounded Unit<64>.");
    }
    #[test]
    fn mid_unit32() {
        let unit: Unit = Unit::new();
        assert_eq!(unit.mid(), 0.5, "Error for mid of DomainBounded Unit<f32>.");
    }
}

mod check_width {
    use tantale::core::{DomainBounded, Int, Nat, Real};

    #[test]
    fn width_real_zero() {
        let real_1 = Real::new(0.0, 10.0);
        assert_eq!(
            real_1.width(),
            10.0,
            "Error for range of DomainBounded Real."
        );
    }
    #[test]
    fn width_nat_zero() {
        let nat_1 = Nat::new(0, 10);
        assert_eq!(nat_1.width(), 10, "Error for range of DomainBounded Nat.");
    }
    #[test]
    fn width_int_zero() {
        let int_1 = Int::new(0, 10);
        assert_eq!(int_1.width(), 10, "Error for range of DomainBounded Int.");
    }
    #[test]
    fn width_real_nzero() {
        let real_1 = Real::new(1.0, 11.0);
        assert_eq!(
            real_1.width(),
            10.0,
            "Error for range of DomainBounded Real."
        );
    }
    #[test]
    fn width_nat_nzero() {
        let nat_1 = Nat::new(1, 11);
        assert_eq!(
            nat_1.width(),
            10,
            "Error for odd range of DomainBounded Nat."
        );
    }
    #[test]
    fn width_int_nzero() {
        let int_1 = Int::new(1, 11);
        assert_eq!(
            int_1.width(),
            10,
            "Error for odd range of DomainBounded Int."
        );
    }
}

mod check_domtype {
    use tantale::core::{Bool, Cat, Domain, Int, Nat, Real, Unit};

    #[test]
    fn tests_real_typedom() {
        trait IsF64 {}
        impl IsF64 for f64 {}
        fn check_type<T: Domain>() -> bool
        where
            T::TypeDom: IsF64,
        {
            true
        }
        assert!(check_type::<Real>(), "Real does not have a f64 TypeDom");
    }
    #[test]
    fn tests_nat_typedom() {
        trait IsUsize {}
        impl IsUsize for u64 {}
        fn check_type<T: Domain>() -> bool
        where
            T::TypeDom: IsUsize,
        {
            true
        }
        assert!(check_type::<Nat>(), "Nat does not have a usize TypeDom");
    }

    #[test]
    fn tests_int_typedom() {
        trait IsIsize {}
        impl IsIsize for i64 {}
        fn check_type<T: Domain>() -> bool
        where
            T::TypeDom: IsIsize,
        {
            true
        }
        assert!(check_type::<Int>(), "Nat does not have a usize TypeDom");
    }

    #[test]
    fn isbool() {
        trait IsBool {}
        impl IsBool for bool {}
        fn check_type<T: Domain>() -> bool
        where
            T::TypeDom: IsBool,
        {
            true
        }
        assert!(check_type::<Bool>(), "Bool does not have a bool TypeDom");
    }

    #[test]
    fn isstr() {
        trait IsStrRef {}
        impl IsStrRef for &str {}
        fn check_type<T: Domain>()
        where
            T::TypeDom: for<'b> IsStrRef,
        {
            assert!(true, "Cat does not have a &str TypeDom.");
        }
        check_type::<Cat>();
    }
    #[test]
    fn tests_unit_typedom() {
        trait IsFloat {}
        impl IsFloat for f64 {}
        fn check_type<T: Domain>() -> bool
        where
            T::TypeDom: IsFloat,
        {
            true
        }
        assert!(check_type::<Unit>(), "Unit does not have a f64 TypeDom");
    }
}

mod check_default_sampler {
    use rand;
    use tantale::core::{Bool, Cat, Domain, Int, Nat, Real, Unit};
    #[test]
    fn sampler_real() {
        let mut rng = rand::rng();
        let real_1 = Real::new(0.0, 10.0);
        assert!(
            real_1.is_in(&real_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_nat() {
        let mut rng = rand::rng();
        let nat_1 = Nat::new(0, 10);
        assert!(
            nat_1.is_in(&nat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_int() {
        let mut rng = rand::rng();
        let int_1 = Int::new(0, 10);
        assert!(
            int_1.is_in(&int_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Int"
        );
    }
    #[test]
    fn sampler_bool() {
        let mut rng = rand::rng();
        let bool_1 = Bool::new();
        assert!(
            bool_1.is_in(&bool_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_cat() {
        let mut rng = rand::rng();
        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
        let cat_1 = Cat::new(&ACTIVATION);
        assert!(
            cat_1.is_in(&cat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_unit() {
        let mut rng = rand::rng();
        let unit_1: Unit = Unit::new();
        assert!(
            unit_1.is_in(&unit_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Unit"
        );
    }
}

mod check_default_sampler_base {
    use rand;
    use tantale::core::{Bool, Cat, Domain, Int, Nat, Real, Unit};
    use tantale::Mixed;

    #[derive(Mixed, PartialEq)]
    pub enum BaseDom {
        Real(Real),
        Int(Int),
        Nat(Nat),
        Bool(Bool),
        Cat(Cat),
        Unit(Unit),
    }

    #[test]
    fn sampler_real() {
        let mut rng = rand::rng();
        let real_1 = BaseDom::Real(Real::new(0.0, 10.0));
        assert!(
            real_1.is_in(&real_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_nat() {
        let mut rng = rand::rng();
        let nat_1 = BaseDom::Nat(Nat::new(0, 10));
        assert!(
            nat_1.is_in(&nat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_int() {
        let mut rng = rand::rng();
        let int_1 = BaseDom::Int(Int::new(0, 10));
        assert!(
            int_1.is_in(&int_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Int"
        );
    }
    #[test]
    fn sampler_bool() {
        let mut rng = rand::rng();
        let bool_1 = BaseDom::Bool(Bool::new());
        assert!(
            bool_1.is_in(&bool_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_cat() {
        let mut rng = rand::rng();
        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
        let cat_1 = BaseDom::Cat(Cat::new(&ACTIVATION));
        assert!(
            cat_1.is_in(&cat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Real"
        );
    }
    #[test]
    fn sampler_unit() {
        let mut rng = rand::rng();
        let unit_1 = BaseDom::Unit(Unit::new());
        assert!(
            unit_1.is_in(&unit_1.sample(&mut rng)),
            "Error while sampling with the default sampler of Unit"
        );
    }
}
