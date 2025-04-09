use super::init_dom::*;

#[test]
fn real_into_int() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_int_2();

    let point = 5.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Real to Int");
    assert_eq!(mapped, 90, "Mapping middle of Real to Int does not match")
}
#[test]
fn real_into_int_lower() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_int_2();

    let point = 0.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Real to Int");
    assert_eq!(
        mapped, 80,
        "Mapping lower bound of Real to Int does not match"
    )
}
#[test]
fn real_into_int_upper() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_int_2();

    let point = 10.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Int");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Real to Int does not match"
    )
}

// NAT to Int

#[test]
fn nat_into_int() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_int_2();

    let point = 5;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Real to Int");
    assert_eq!(mapped, 90, "Mapping middle of Real to Int does not match")
}
#[test]
fn nat_into_int_lower() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_int_2();

    let point = 0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Real to Int");
    assert_eq!(
        mapped, 80,
        "Mapping lower bound of Real to Int does not match"
    )
}
#[test]
fn nat_into_int_upper() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_int_2();

    let point = 10;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Int");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Real to Int does not match"
    )
}

// INT to Int

#[test]
fn int_into_int() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_int_2();

    let point = 5;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Int to Int");
    assert_eq!(mapped, 90, "Mapping middle of Int to Int does not match")
}
#[test]
fn int_into_int_lower() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_int_2();

    let point = 0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Int to Int");
    assert_eq!(
        mapped, 80,
        "Mapping lower bound of Int to Int does not match"
    )
}
#[test]
fn int_into_int_upper() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_int_2();

    let point = 10;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Int to Int");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Int to Int does not match"
    )
}

// BOOL to Int
#[test]
fn bool_into_int_false() {
    let domain_1 = get_domain_bool();
    let domain_2 = get_domain_int_2();

    let point = false;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Nat to Int");
    assert_eq!(
        mapped, 80,
        "Mapping lower bound of Nat to Int does not match"
    )
}
#[test]
fn bool_into_int_true() {
    let domain_1 = get_domain_bool();
    let domain_2 = get_domain_int_2();

    let point = true;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Nat to Int");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Nat to Int does not match"
    )
}

// CAT to Nat

#[test]
fn cat_into_int() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_int_2();

    let point = "tanh";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Cat to Nat");
    assert_eq!(mapped, 90, "Mapping middle of Cat to Nat does not match")
}
#[test]
fn cat_into_int_lower() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_int_2();

    let point = "relu";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Cat to Nat");
    assert_eq!(
        mapped, 80,
        "Mapping lower bound of Cat to Nat does not match"
    )
}
#[test]
fn cat_into_int_upper() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_int_2();

    let point = "sigmoid";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Real");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Cat to Nat does not match"
    )
}

#[test]
#[should_panic(
    expected = "Error in mapping upper bound from Cat to Int: Input out of bounds, pineapple not in {relu, tanh, sigmoid}."
)]
fn cat_into_int_oob() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_int_2();

    let point = "pineapple";

    let _mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Cat to Int");
}
