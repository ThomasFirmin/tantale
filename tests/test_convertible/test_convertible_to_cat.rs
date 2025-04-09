use super::init_dom::*;

#[test]
fn real_into_cat() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_cat_2();

    let point = 5.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Real to Cat");
    assert_eq!(
        mapped, "tanh",
        "Mapping middle of Real to Cat does not match"
    )
}
#[test]
fn real_into_cat_lower() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_cat_2();

    let point = 0.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Real to Cat");
    assert_eq!(
        mapped, "relu",
        "Mapping lower bound of Real to Cat does not match"
    )
}
#[test]
fn real_into_cat_upper() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_cat_2();

    let point = 10.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Cat");
    assert_eq!(
        mapped, "sigmoid",
        "Mapping upper bound of Real to Cat does not match"
    )
}

// NAT to Cat

#[test]
fn nat_into_cat() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_cat_2();

    let point = 5;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Real to Cat");
    assert_eq!(
        mapped, "tanh",
        "Mapping middle of Real to Cat does not match"
    )
}
#[test]
fn nat_into_cat_lower() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_cat_2();

    let point = 0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Real to Cat");
    assert_eq!(
        mapped, "relu",
        "Mapping lower bound of Real to Cat does not match"
    )
}
#[test]
fn nat_into_cat_upper() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_cat_2();

    let point = 10;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Cat");
    assert_eq!(
        mapped, "sigmoid",
        "Mapping upper bound of Real to Cat does not match"
    )
}

// INT to Cat

#[test]
fn int_into_cat() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_cat_2();

    let point = 5;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Int to Cat");
    assert_eq!(
        mapped, "tanh",
        "Mapping middle of Int to Cat does not match"
    )
}
#[test]
fn int_into_cat_lower() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_cat_2();

    let point = 0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Int to Cat");
    assert_eq!(
        mapped, "relu",
        "Mapping lower bound of Int to Cat does not match"
    )
}
#[test]
fn int_into_cat_upper() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_cat_2();

    let point = 10;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Int to Cat");
    assert_eq!(
        mapped, "sigmoid",
        "Mapping upper bound of Int to Cat does not match"
    )
}

// BOOL to Cat
#[test]
fn bool_into_cat_false() {
    let domain_1 = get_domain_bool();
    let domain_2 = get_domain_cat_2();

    let point = false;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Nat to Cat");
    assert_eq!(
        mapped, "relu",
        "Mapping lower bound of Nat to Cat does not match"
    )
}
#[test]
fn bool_into_cat_true() {
    let domain_1 = get_domain_bool();
    let domain_2 = get_domain_cat_2();

    let point = true;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Nat to Cat");
    assert_eq!(
        mapped, "sigmoid",
        "Mapping upper bound of Nat to Cat does not match"
    )
}
