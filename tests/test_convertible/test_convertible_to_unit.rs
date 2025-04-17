use super::init_dom::*;

#[test]
fn real_into_unit() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_unit_2();

    let point = 5.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Real to Unit");
    assert_eq!(
        mapped, 0.5,
        "Mapping middle of Real to Unit does not match"
    )
}
#[test]
fn real_into_unit_lower() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_unit_2();

    let point = 0.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Real to Unit");
    assert_eq!(
        mapped, 0.0,
        "Mapping lower bound of Real to Unit does not match"
    )
}
#[test]
fn real_into_unit_upper() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_unit_2();

    let point = 10.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Unit");
    assert_eq!(
        mapped, 1.0,
        "Mapping upper bound of Real to Unit does not match"
    )
}

#[test]
#[should_panic]
fn real_into_unit_oob() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_unit_2();

    let point = 11.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Unit");
    assert_eq!(
        mapped, 1.0,
        "Mapping upper bound of Real to Unit does not match"
    )
}

// NAT to Unit

#[test]
fn nat_into_unit() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_unit_2();

    let point = 6;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Real to Unit");
    assert_eq!(
        mapped, 0.5,
        "Mapping middle of Real to Unit does not match"
    )
}
#[test]
fn nat_into_unit_lower() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_unit_2();

    let point = 1;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Real to Unit");
    assert_eq!(
        mapped, 0.0,
        "Mapping lower bound of Real to Unit does not match"
    )
}
#[test]
fn nat_into_unit_upper() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_unit_2();

    let point = 11;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Unit");
    assert_eq!(
        mapped, 1.0,
        "Mapping upper bound of Real to Unit does not match"
    )
}

#[test]
#[should_panic]
fn nat_into_unit_oob() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_unit_2();

    let point = 12;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Nat to Unit");
    assert_eq!(
        mapped, 1.0,
        "Mapping upper bound of Nat to Unit does not match"
    )
}

// INT to Unit

#[test]
fn int_into_unit() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_unit_2();

    let point = 5;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Int to Unit");
    assert_eq!(mapped, 0.5, "Mapping middle of Int to Unit does not match")
}
#[test]
fn int_into_unit_lower() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_unit_2();

    let point = 0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Int to Unit");
    assert_eq!(
        mapped, 0.0,
        "Mapping lower bound of Int to Unit does not match"
    )
}
#[test]
fn int_into_unit_upper() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_unit_2();

    let point = 10;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Int to Unit");
    assert_eq!(
        mapped, 1.0,
        "Mapping upper bound of Int to Unit does not match"
    )
}

#[test]
#[should_panic]
fn int_into_unit_oob() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_unit_2();

    let point = 11;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Int to Unit");
    assert_eq!(
        mapped, 1.0,
        "Mapping upper bound of Int to Unit does not match"
    )
}

// BOOL to Unit
#[test]
fn bool_into_unit_false() {
    let domain_1 = get_domain_bool();
    let domain_2 = get_domain_unit_2();

    let point = false;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Nat to Unit");
    assert_eq!(
        mapped, 0.0,
        "Mapping lower bound of Nat to Unit does not match"
    )
}
#[test]
fn bool_into_unit_true() {
    let domain_1 = get_domain_bool();
    let domain_2 = get_domain_unit_2();

    let point = true;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Nat to Unit");
    assert_eq!(
        mapped, 1.0,
        "Mapping upper bound of Nat to Unit does not match"
    )
}

// CAT to Unit

#[test]
fn cat_into_unit() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_unit_2();

    let point = "tanh";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Cat to Unit");
    assert_eq!(mapped, 0.5, "Mapping middle of Cat to Unit does not match")
}
#[test]
fn cat_into_unit_lower() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_unit_2();

    let point = "relu";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Cat to Unit");
    assert_eq!(
        mapped, 0.0,
        "Mapping lower bound of Cat to Unit does not match"
    )
}
#[test]
fn cat_into_unit_upper() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_unit_2();

    let point = "sigmoid";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Cat to Unit");
    assert_eq!(
        mapped, 1.0,
        "Mapping upper bound of Cat to Unit does not match"
    )
}

#[test]
#[should_panic]
fn cat_into_unit_oob() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_unit_2();

    let point = "pineapple";

    let _mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Cat to Unit");
}