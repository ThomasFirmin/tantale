use super::init_dom::*;

#[test]
fn real_into_bool() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_bool_2();

    let point = 5.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Real to Bool");
    assert!(!mapped, "Mapping middle of Real to Bool does not match")
}
#[test]
fn real_into_bool_lower() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_bool_2();

    let point = 0.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Real to Bool");
    assert!(
        !mapped,
        "Mapping lower bound of Real to Bool does not match"
    )
}
#[test]
fn real_into_bool_upper() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_bool_2();

    let point = 10.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Bool");
    assert!(mapped, "Mapping upper bound of Real to Bool does not match")
}
#[test]
#[should_panic]
fn real_into_bool_oob() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_bool_2();

    let point = 11.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Bool");
    assert_eq!(
        mapped, true,
        "Mapping upper bound of Real to Bool does not match"
    )
}

// NAT to Bool

#[test]
fn nat_into_bool() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_bool_2();

    let point = 6;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Real to Bool");
    assert!(!mapped, "Mapping middle of Real to Bool does not match")
}
#[test]
fn nat_into_bool_lower() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_bool_2();

    let point = 1;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Real to Bool");
    assert!(
        !mapped,
        "Mapping lower bound of Real to Bool does not match"
    )
}
#[test]
fn nat_into_bool_upper() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_bool_2();

    let point = 11;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Bool");
    assert!(mapped, "Mapping upper bound of Real to Bool does not match")
}
#[test]
#[should_panic]
fn nat_into_bool_oob() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_bool_2();

    let point = 12;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Nat to Bool");
    assert_eq!(
        mapped, true,
        "Mapping upper bound of Nat to Bool does not match"
    )
}

// INT to Bool

#[test]
fn int_into_bool() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_bool_2();

    let point = 5;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Int to Bool");
    assert!(!mapped, "Mapping middle of Int to Bool does not match")
}
#[test]
fn int_into_bool_lower() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_bool_2();

    let point = 0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Int to Bool");
    assert!(!mapped, "Mapping lower bound of Int to Bool does not match")
}
#[test]
fn int_into_bool_upper() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_bool_2();

    let point = 10;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Int to Bool");
    assert!(mapped, "Mapping upper bound of Int to Bool does not match")
}
#[test]
#[should_panic]
fn int_into_bool_oob() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_bool_2();

    let point = 11;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Int to Bool");
    assert!(
        mapped,
        "Mapping upper bound of Int to Bool does not match"
    )
}


#[test]
fn unit_into_bool() {
    let domain_1 = get_domain_unit();
    let domain_2 = get_domain_bool_2();

    let point = 0.5;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Unit to Bool");
    assert!(!mapped, "Mapping middle of Unit to Bool does not match")
}
#[test]
fn unit_into_bool_lower() {
    let domain_1 = get_domain_unit();
    let domain_2 = get_domain_bool_2();

    let point = 0.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Unit to Bool");
    assert!(
        !mapped,
        "Mapping lower bound of Unit to Bool does not match"
    )
}
#[test]
fn unit_into_bool_upper() {
    let domain_1 = get_domain_unit();
    let domain_2 = get_domain_bool_2();

    let point = 1.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Unit to Bool");
    assert!(mapped, "Mapping upper bound of Unit to Bool does not match")
}
#[test]
#[should_panic]
fn unit_into_bool_oob() {
    let domain_1 = get_domain_unit();
    let domain_2 = get_domain_bool_2();

    let point = 1.1;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Unit to Bool");
    assert_eq!(
        mapped, true,
        "Mapping upper bound of Unit to Bool does not match"
    )
}