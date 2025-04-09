use super::init_dom::*;

#[test]
fn real_into_real() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_real_2();

    let point = 5.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Real to Real");
    assert_eq!(
        mapped, 90.0,
        "Mapping middle of Real to Real does not match"
    )
}
#[test]
fn real_into_real_lower() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_real_2();

    let point = 0.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Real to Real");
    assert_eq!(
        mapped, 80.0,
        "Mapping lower bound of Real to Real does not match"
    )
}
#[test]
fn real_into_real_upper() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_real_2();

    let point = 10.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Real");
    assert_eq!(
        mapped, 100.0,
        "Mapping upper bound of Real to Real does not match"
    )
}

#[test]
#[should_panic(
    expected = "Error in mapping upper bound from Real to Real: Input out of bounds, 11 not in [0,10]."
)]
fn real_into_real_oob() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_real_2();

    let point = 11.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Real");
    assert_eq!(
        mapped, 100.0,
        "Mapping upper bound of Real to Real does not match"
    )
}

// NAT TO REAL

#[test]
fn nat_into_real() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_real_2();

    let point = 5;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Real to Real");
    assert_eq!(
        mapped, 90.0,
        "Mapping middle of Real to Real does not match"
    )
}
#[test]
fn nat_into_real_lower() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_real_2();

    let point = 0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Real to Real");
    assert_eq!(
        mapped, 80.0,
        "Mapping lower bound of Real to Real does not match"
    )
}
#[test]
fn nat_into_real_upper() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_real_2();

    let point = 10;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Real");
    assert_eq!(
        mapped, 100.0,
        "Mapping upper bound of Real to Real does not match"
    )
}

#[test]
#[should_panic(
    expected = "Error in mapping upper bound from Nat to Real: Input out of bounds, 11 not in [0,10]."
)]
fn nat_into_real_oob() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_real_2();

    let point = 11;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Nat to Real");
    assert_eq!(
        mapped, 100.0,
        "Mapping upper bound of Nat to Real does not match"
    )
}

// INT TO REAL

#[test]
fn int_into_real() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_real_2();

    let point = 5;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Int to Real");
    assert_eq!(mapped, 90.0, "Mapping middle of Int to Real does not match")
}
#[test]
fn int_into_real_lower() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_real_2();

    let point = 0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Int to Real");
    assert_eq!(
        mapped, 80.0,
        "Mapping lower bound of Int to Real does not match"
    )
}
#[test]
fn int_into_real_upper() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_real_2();

    let point = 10;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Int to Real");
    assert_eq!(
        mapped, 100.0,
        "Mapping upper bound of Int to Real does not match"
    )
}

#[test]
#[should_panic(
    expected = "Error in mapping upper bound from Int to Real: Input out of bounds, 11 not in [0,10]."
)]
fn int_into_real_oob() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_real_2();

    let point = 11;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Int to Real");
    assert_eq!(
        mapped, 100.0,
        "Mapping upper bound of Int to Real does not match"
    )
}

// BOOL TO REAL
#[test]
fn bool_into_real_false() {
    let domain_1 = get_domain_bool();
    let domain_2 = get_domain_real_2();

    let point = false;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Nat to Real");
    assert_eq!(
        mapped, 80.0,
        "Mapping lower bound of Nat to Real does not match"
    )
}
#[test]
fn bool_into_real_true() {
    let domain_1 = get_domain_bool();
    let domain_2 = get_domain_real_2();

    let point = true;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Nat to Real");
    assert_eq!(
        mapped, 100.0,
        "Mapping upper bound of Nat to Real does not match"
    )
}

// CAT to REAL

#[test]
fn cat_into_real() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_real_2();

    let point = "tanh";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Cat to Real");
    assert_eq!(mapped, 90.0, "Mapping middle of Cat to Real does not match")
}
#[test]
fn cat_into_real_lower() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_real_2();

    let point = "relu";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Cat to Real");
    assert_eq!(
        mapped, 80.0,
        "Mapping lower bound of Cat to Real does not match"
    )
}
#[test]
fn cat_into_real_upper() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_real_2();

    let point = "sigmoid";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Cat to Real");
    assert_eq!(
        mapped, 100.0,
        "Mapping upper bound of Cat to Real does not match"
    )
}

#[test]
#[should_panic(
    expected = "Error in mapping upper bound from Cat to Real: Input out of bounds, pineapple not in {relu, tanh, sigmoid}."
)]
fn cat_into_real_oob() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_real_2();

    let point = "pineapple";

    let _mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Cat to Real");
}
