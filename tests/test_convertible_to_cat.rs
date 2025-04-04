mod check_into_cat {

    use tantale::core::convertible::Convertible;
    use tantale::core::domain::{Bool, Cat, Int, Nat, Real};
    fn get_domain_real() -> Real {
        return Real::new(0.0, 10.0).expect("Error while creating input Real domain");
    }
    fn get_domain_nat() -> Nat {
        return Nat::new(0, 10).expect("Error while creating input Nat domain");
    }
    fn get_domain_int() -> Int {
        return Int::new(0, 10).expect("Error while creating input Int domain");
    }
    fn get_domain_bool() -> Bool {
        return Bool::new().expect("Error while creating input Bool domain");
    }
    fn get_domain_cat<'a>() -> Cat<'a, 3> {
        let activation = ["relu", "tanh", "sigmoid"];
        return Cat::new(activation).expect("Error while creating Cat");
    }
    fn get_domain_2<'a>() -> Cat<'a, 3> {
        let activation = ["relu", "tanh", "sigmoid"];
        return Cat::new(activation).expect("Error while creating Cat");
    }

    #[test]
    fn test_real_into_cat() {
        let domain_1 = get_domain_real();
        let domain_2 = get_domain_2();

        let point = 5.0;

        let mapped = domain_1
            .to_cat(&point, &domain_2)
            .expect("Error in mapping middle from Real to Cat");
        assert_eq!(
            mapped, "tanh",
            "Mapping middle of Real to Cat does not match"
        )
    }
    #[test]
    fn test_real_into_cat_lower() {
        let domain_1 = get_domain_real();
        let domain_2 = get_domain_2();

        let point = 0.0;

        let mapped = domain_1
            .to_cat(&point, &domain_2)
            .expect("Error in mapping lower bound from Real to Cat");
        assert_eq!(
            mapped, "relu",
            "Mapping lower bound of Real to Cat does not match"
        )
    }
    #[test]
    fn test_real_into_cat_upper() {
        let domain_1 = get_domain_real();
        let domain_2 = get_domain_2();

        let point = 10.0;

        let mapped = domain_1
            .to_cat(&point, &domain_2)
            .expect("Error in mapping upper bound from Real to Cat");
        assert_eq!(
            mapped, "sigmoid",
            "Mapping upper bound of Real to Cat does not match"
        )
    }

    // NAT to Cat

    #[test]
    fn test_nat_into_cat() {
        let domain_1 = get_domain_nat();
        let domain_2 = get_domain_2();

        let point = 5;

        let mapped = domain_1
            .to_cat(&point, &domain_2)
            .expect("Error in mapping middle from Real to Cat");
        assert_eq!(
            mapped, "tanh",
            "Mapping middle of Real to Cat does not match"
        )
    }
    #[test]
    fn test_nat_into_cat_lower() {
        let domain_1 = get_domain_nat();
        let domain_2 = get_domain_2();

        let point = 0;

        let mapped = domain_1
            .to_cat(&point, &domain_2)
            .expect("Error in mapping lower bound from Real to Cat");
        assert_eq!(
            mapped, "relu",
            "Mapping lower bound of Real to Cat does not match"
        )
    }
    #[test]
    fn test_nat_into_cat_upper() {
        let domain_1 = get_domain_nat();
        let domain_2 = get_domain_2();

        let point = 10;

        let mapped = domain_1
            .to_cat(&point, &domain_2)
            .expect("Error in mapping upper bound from Real to Cat");
        assert_eq!(
            mapped, "sigmoid",
            "Mapping upper bound of Real to Cat does not match"
        )
    }

    // INT to Cat

    #[test]
    fn test_int_into_cat() {
        let domain_1 = get_domain_int();
        let domain_2 = get_domain_2();

        let point = 5;

        let mapped = domain_1
            .to_cat(&point, &domain_2)
            .expect("Error in mapping middle from Int to Cat");
        assert_eq!(
            mapped, "tanh",
            "Mapping middle of Int to Cat does not match"
        )
    }
    #[test]
    fn test_int_into_cat_lower() {
        let domain_1 = get_domain_int();
        let domain_2 = get_domain_2();

        let point = 0;

        let mapped = domain_1
            .to_cat(&point, &domain_2)
            .expect("Error in mapping lower bound from Int to Cat");
        assert_eq!(
            mapped, "relu",
            "Mapping lower bound of Int to Cat does not match"
        )
    }
    #[test]
    fn test_int_into_cat_upper() {
        let domain_1 = get_domain_int();
        let domain_2 = get_domain_2();

        let point = 10;

        let mapped = domain_1
            .to_cat(&point, &domain_2)
            .expect("Error in mapping upper bound from Int to Cat");
        assert_eq!(
            mapped, "sigmoid",
            "Mapping upper bound of Int to Cat does not match"
        )
    }

    // BOOL to Cat
    #[test]
    fn test_bool_into_cat_false() {
        let domain_1 = get_domain_bool();
        let domain_2 = get_domain_2();

        let point = false;

        let mapped = domain_1
            .to_cat(&point, &domain_2)
            .expect("Error in mapping lower bound from Nat to Cat");
        assert_eq!(
            mapped, "relu",
            "Mapping lower bound of Nat to Cat does not match"
        )
    }
    #[test]
    fn test_bool_into_cat_true() {
        let domain_1 = get_domain_bool();
        let domain_2 = get_domain_2();

        let point = true;

        let mapped = domain_1
            .to_cat(&point, &domain_2)
            .expect("Error in mapping upper bound from Nat to Cat");
        assert_eq!(
            mapped, "sigmoid",
            "Mapping upper bound of Nat to Cat does not match"
        )
    }

    // CAT to Cat

    #[test]
    #[should_panic(expected = "Error in mapping from Cat to Cat")]
    fn cat_into_cat() {
        let domain_1 = get_domain_cat();
        let domain_2 = get_domain_2();

        let point = "tanh";

        let mapped = domain_1
            .to_cat(&point, &domain_2)
            .expect("Error in mapping from Cat to Cat");
        assert_eq!(
            mapped, "tanh",
            "Mapping middle of Cat to Cat does not match"
        )
    }
}
