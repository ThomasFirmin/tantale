use super::init_dom::*;

#[test]
fn real_into_nat() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_nat_2();

    let point = 5.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Real to Nat");
    assert_eq!(mapped, 90, "Mapping middle of Real to Nat does not match")
}
#[test]
fn real_into_nat_lower() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_nat_2();

    let point = 0.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Real to Nat");
    assert_eq!(
        mapped, 80,
        "Mapping lower bound of Real to Nat does not match"
    )
}
#[test]
fn real_into_nat_upper() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_nat_2();

    let point = 10.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Nat");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Real to Nat does not match"
    )
}

#[test]
#[should_panic]
fn real_into_nat_oob() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_nat_2();

    let point = 11.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Nat");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Real to Nat does not match"
    )
}

// NAT to Nat

#[test]
fn nat_into_nat() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_nat_2();

    let point = 6;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Real to Nat");
    assert_eq!(mapped, 90, "Mapping middle of Real to Nat does not match")
}
#[test]
fn nat_into_nat_lower() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_nat_2();

    let point = 1;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Real to Nat");
    assert_eq!(
        mapped, 80,
        "Mapping lower bound of Real to Nat does not match"
    )
}
#[test]
fn nat_into_nat_upper() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_nat_2();

    let point = 11;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Nat");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Real to Nat does not match"
    )
}

#[test]
#[should_panic]
fn nat_into_nat_oob() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_nat_2();

    let point = 12;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Nat to Nat");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Nat to Nat does not match"
    )
}

// INT to Nat

#[test]
fn int_into_nat() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_nat_2();

    let point = 5;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Int to Nat");
    assert_eq!(mapped, 90, "Mapping middle of Int to Nat does not match")
}
#[test]
fn int_into_nat_lower() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_nat_2();

    let point = 0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Int to Nat");
    assert_eq!(
        mapped, 80,
        "Mapping lower bound of Int to Nat does not match"
    )
}
#[test]
fn int_into_nat_upper() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_nat_2();

    let point = 10;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Int to Nat");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Int to Nat does not match"
    )
}

#[test]
#[should_panic]
fn int_into_nat_oob() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_nat_2();

    let point = 11;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Int to Nat");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Int to Nat does not match"
    )
}

// BOOL to Nat
#[test]
fn bool_into_nat_false() {
    let domain_1 = get_domain_bool();
    let domain_2 = get_domain_nat_2();

    let point = false;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Nat to Nat");
    assert_eq!(
        mapped, 80,
        "Mapping lower bound of Nat to Nat does not match"
    )
}
#[test]
fn bool_into_nat_true() {
    let domain_1 = get_domain_bool();
    let domain_2 = get_domain_nat_2();

    let point = true;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Nat to Nat");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Nat to Nat does not match"
    )
}

// CAT to Nat

#[test]
fn cat_into_nat() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_nat_2();

    let point = "tanh";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Cat to Nat");
    assert_eq!(mapped, 90, "Mapping middle of Cat to Nat does not match")
}
#[test]
fn cat_into_nat_lower() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_nat_2();

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
fn cat_into_nat_upper() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_nat_2();

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
#[should_panic]
fn cat_into_nat_oob() {
    let domain_1 = get_domain_cat();
    let domain_2 = get_domain_nat_2();

    let point = "pineapple";

    let _mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Cat to Nat");
}

#[test]
fn unit_into_nat() {
    let domain_1 = get_domain_unit();
    let domain_2 = get_domain_nat_2();

    let point = 0.5;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Unit to Nat");
    assert_eq!(mapped, 90, "Mapping middle of Unit to Nat does not match")
}
#[test]
fn unit_into_nat_lower() {
    let domain_1 = get_domain_unit();
    let domain_2 = get_domain_nat_2();

    let point = 0.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Unit to Nat");
    assert_eq!(
        mapped, 80,
        "Mapping lower bound of Unit to Nat does not match"
    )
}
#[test]
fn unit_into_nat_upper() {
    let domain_1 = get_domain_unit();
    let domain_2 = get_domain_nat_2();

    let point = 1.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Unit to Nat");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Unit to Nat does not match"
    )
}

#[test]
#[should_panic]
fn unit_into_nat_oob() {
    let domain_1 = get_domain_unit();
    let domain_2 = get_domain_nat_2();

    let point = 1.1;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Unit to Nat");
    assert_eq!(
        mapped, 100,
        "Mapping upper bound of Unit to Nat does not match"
    )
}
