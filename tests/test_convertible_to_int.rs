mod check_into_int {
    
    use tantale::core::convertible::Convertible;
    use tantale::core::domain::{Real, Nat, Int, Bool};
    fn get_domain_real()->Real{
        return Real::new(0.0, 10.0).expect("Error while creating input Real domain");
    }
    fn get_domain_nat()->Nat{
        return Nat::new(0, 10).expect("Error while creating input Nat domain");
    }
    fn get_domain_int()->Int{
        return Int::new(0, 10).expect("Error while creating input Int domain");
    }
    fn get_domain_bool()->Bool{
        return Bool::new().expect("Error while creating input Bool domain");
    }
    fn get_domain_2()->Int{
        return Int::new(80, 100).expect("Error while creating output domain");
    }

    #[test]
    fn test_real_into_int() {

        let domain_1 = get_domain_real();
        let domain_2 = get_domain_2();

        let point = 5.0;

        let mapped = domain_1
            .to_int(&point, &domain_2)
            .expect("Error in mapping middle from Real to Int");
        assert_eq!(
            mapped, 90,
            "Mapping middle of Real to Int does not match"
        )
    }
    #[test]
    fn test_real_into_int_lower() {
        let domain_1 = get_domain_real();
        let domain_2 = get_domain_2();

        let point = 0.0;

        let mapped = domain_1
            .to_int(&point, &domain_2)
            .expect("Error in mapping lower bound from Real to Int");
        assert_eq!(
            mapped, 80,
            "Mapping lower bound of Real to Int does not match"
        )
    }
    #[test]
    fn test_real_into_int_upper() {
        let domain_1 = get_domain_real();
        let domain_2 = get_domain_2();

        let point = 10.0;

        let mapped = domain_1
            .to_int(&point, &domain_2)
            .expect("Error in mapping upper bound from Real to Int");
        assert_eq!(
            mapped, 100,
            "Mapping upper bound of Real to Int does not match"
        )
    }

    // NAT to Int

    #[test]
    fn test_nat_into_int() {
        let domain_1 = get_domain_nat();
        let domain_2 = get_domain_2();

        let point = 5;

        let mapped = domain_1
            .to_int(&point, &domain_2)
            .expect("Error in mapping middle from Real to Int");
        assert_eq!(
            mapped, 90,
            "Mapping middle of Real to Int does not match"
        )
    }
    #[test]
    fn test_nat_into_int_lower() {

        let domain_1 = get_domain_nat();
        let domain_2 = get_domain_2();

        let point = 0;

        let mapped = domain_1
            .to_int(&point, &domain_2)
            .expect("Error in mapping lower bound from Real to Int");
        assert_eq!(
            mapped, 80,
            "Mapping lower bound of Real to Int does not match"
        )
    }
    #[test]
    fn test_nat_into_int_upper() {
        let domain_1 = get_domain_nat();
        let domain_2 = get_domain_2();

        let point = 10;

        let mapped = domain_1
            .to_int(&point, &domain_2)
            .expect("Error in mapping upper bound from Real to Int");
        assert_eq!(
            mapped, 100,
            "Mapping upper bound of Real to Int does not match"
        )
    }

    // INT to Int

    #[test]
    fn test_int_into_int() {
        let domain_1 = get_domain_int();
        let domain_2 = get_domain_2();

        let point = 5;

        let mapped = domain_1
            .to_int(&point, &domain_2)
            .expect("Error in mapping middle from Int to Int");
        assert_eq!(
            mapped, 90,
            "Mapping middle of Int to Int does not match"
        )
    }
    #[test]
    fn test_int_into_int_lower() {

        let domain_1 = get_domain_int();
        let domain_2 = get_domain_2();

        let point = 0;

        let mapped = domain_1
            .to_int(&point, &domain_2)
            .expect("Error in mapping lower bound from Int to Int");
        assert_eq!(
            mapped, 80,
            "Mapping lower bound of Int to Int does not match"
        )
    }
    #[test]
    fn test_int_into_int_upper() {
        let domain_1 = get_domain_int();
        let domain_2 = get_domain_2();

        let point = 10;

        let mapped = domain_1
            .to_int(&point, &domain_2)
            .expect("Error in mapping upper bound from Int to Int");
        assert_eq!(
            mapped, 100,
            "Mapping upper bound of Int to Int does not match"
        )
    }

    // BOOL to Int
    #[test]
    fn test_bool_into_int_false() {

        let domain_1 = get_domain_bool();
        let domain_2 = get_domain_2();

        let point = false;

        let mapped = domain_1
            .to_int(&point, &domain_2)
            .expect("Error in mapping lower bound from Nat to Int");
        assert_eq!(
            mapped, 80,
            "Mapping lower bound of Nat to Int does not match"
        )
    }
    #[test]
    fn test_bool_into_int_true() {
        let domain_1 = get_domain_bool();
        let domain_2 = get_domain_2();

        let point = true;

        let mapped = domain_1
            .to_int(&point, &domain_2)
            .expect("Error in mapping upper bound from Nat to Int");
        assert_eq!(
            mapped, 100,
            "Mapping upper bound of Nat to Int does not match"
        )
    }

}