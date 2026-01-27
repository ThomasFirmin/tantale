mod check_sampler {
    use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real};
    use tantale::core::sampler::*;

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
}

mod check_sampler_base {
    use tantale::core::domain::{Mixed, Bool, Cat, Domain, Int, Nat, Real, Unit};
    use tantale::core::sampler::*;

    #[test]
    fn sampler_real_base() {
        let mut rng = rand::rng();
        let basedom: Mixed = Real::new(0.0, 10.0, Uniform).into();

        assert!(
            basedom.is_in(&basedom.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Real"
        );
    }
    #[test]
    fn sampler_nat_base() {
        let mut rng = rand::rng();
        let basedom: Mixed = Nat::new(0, 10, Uniform).into();

        assert!(
            basedom.is_in(&basedom.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Nat"
        );
    }
    #[test]
    fn sampler_int_base() {
        let mut rng = rand::rng();
        let basedom: Mixed = Int::new(0, 10, Uniform).into();

        assert!(
            basedom.is_in(&basedom.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Int"
        );
    }
    #[test]
    fn sampler_bool_base() {
        let mut rng = rand::rng();
        let basedom: Mixed = Bool::new(Bernoulli(0.5)).into();

        assert!(
            basedom.is_in(&basedom.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Bool"
        );
    }
    #[test]
    fn sampler_cat_base() {
        let mut rng = rand::rng();
        let basedom: Mixed = Cat::new(["relu", "tanh", "sigmoid"], Uniform).into();

        assert!(
            basedom.is_in(&basedom.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Cat"
        );
    }
    #[test]
    fn sampler_unit_base() {
        let mut rng = rand::rng();
        let basedom: Mixed = Unit::new(Uniform).into();

        assert!(
            basedom.is_in(&basedom.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Unit"
        );
    }
    #[test]
    fn sampler_base_many() {
        let mut rng = rand::rng();
        let basedom_real_1: Mixed = Real::new(0.0, 10.0, Uniform).into();
        let basedom_real_2: Mixed = Real::new(100.0, 1000.0, Uniform).into();

        let basedom_nat_1: Mixed = Nat::new(0, 10, Uniform).into();
        let basedom_nat_2: Mixed = Nat::new(100, 1000, Uniform).into();

        let basedom_int_1: Mixed = Int::new(0, 10, Uniform).into();
        let basedom_int_2: Mixed = Int::new(100, 1000, Uniform).into();

        let basedom_bool_1: Mixed = Bool::new(Bernoulli(0.5)).into();
        let basedom_bool_2: Mixed = Bool::new(Bernoulli(0.5)).into();

        let basedom_cat_1: Mixed = Cat::new(["relu", "tanh", "sigmoid"], Uniform).into();
        let basedom_cat_2: Mixed = Cat::new(["relu", "tanh", "sigmoid"], Uniform).into();

        let basedom_unit_1: Mixed = Unit::new(Uniform).into();
        let basedom_unit_2: Mixed = Unit::new(Uniform).into();

        assert!(
            basedom_real_1.is_in(&basedom_real_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Real"
        );
        assert!(
            basedom_real_2.is_in(&basedom_real_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Real"
        );

        assert!(
            basedom_nat_1.is_in(&basedom_nat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Nat"
        );
        assert!(
            basedom_nat_2.is_in(&basedom_nat_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Nat"
        );

        assert!(
            basedom_int_1.is_in(&basedom_int_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Int"
        );
        assert!(
            basedom_int_2.is_in(&basedom_int_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Int"
        );

        assert!(
            basedom_bool_1.is_in(&basedom_bool_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Bool"
        );
        assert!(
            basedom_bool_2.is_in(&basedom_bool_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Bool"
        );

        assert!(
            basedom_cat_1.is_in(&basedom_cat_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Cat"
        );
        assert!(
            basedom_cat_2.is_in(&basedom_cat_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Cat"
        );

        assert!(
            basedom_unit_1.is_in(&basedom_unit_1.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Unit"
        );
        assert!(
            basedom_unit_2.is_in(&basedom_unit_2.sample(&mut rng)),
            "Error while sampling with the default sampler of BaseDom::Unit"
        );
    }
}
