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

// BOOL to Bool
#[test]
fn bool_into_bool_false() {
    let domain_1 = get_domain_bool();
    let domain_2 = get_domain_bool_2();

    let point = false;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Nat to Bool");
    assert!(!mapped, "Mapping lower bound of Nat to Bool does not match")
}
#[test]
fn bool_into_bool_true() {
    let domain_1 = get_domain_bool();
    let domain_2 = get_domain_bool_2();

    let point = true;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Nat to Bool");
    assert!(mapped, "Mapping upper bound of Nat to Bool does not match")
}
