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
#[test]
#[should_panic]
fn real_intocatt_oob() {
    let domain_1 = get_domain_real();
    let domain_2 = get_domain_cat_2();

    let point = 11.0;
    let check = "sigmoid";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Cat");
    assert_eq!(
        mapped, check,
        "Mapping upper bound of Real to Cat does not match"
    )
}

// NAT to Cat

#[test]
fn nat_into_cat() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_cat_2();

    let point = 6;

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

    let point = 1;

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

    let point = 11;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Real to Cat");
    assert_eq!(
        mapped, "sigmoid",
        "Mapping upper bound of Real to Cat does not match"
    )
}
#[test]
#[should_panic]
fn nat_into_cat_oob() {
    let domain_1 = get_domain_nat();
    let domain_2 = get_domain_cat_2();

    let point = 12;
    let check = "sigmoid";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Nat to Cat");
    assert_eq!(
        mapped, check,
        "Mapping upper bound of Nat to Cat does not match"
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
#[test]
#[should_panic]
fn int_into_cat_oob() {
    let domain_1 = get_domain_int();
    let domain_2 = get_domain_cat_2();

    let point = 11;
    let check = "sigmoid";

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Int to Cat");
    assert_eq!(
        mapped, check,
        "Mapping upper bound of Int to Cat does not match"
    )
}


#[test]
fn unit_into_cat() {
    let domain_1 = get_domain_unit();
    let domain_2 = get_domain_cat_2();

    let point = 0.5;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping middle from Unit to Cat");
    assert_eq!(
        mapped, "tanh",
        "Mapping middle of Unit to Cat does not match"
    )
}
#[test]
fn unit_into_cat_lower() {
    let domain_1 = get_domain_unit();
    let domain_2 = get_domain_cat_2();

    let point = 0.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping lower bound from Unit to Cat");
    assert_eq!(
        mapped, "relu",
        "Mapping lower bound of Unit to Cat does not match"
    )
}
#[test]
fn unit_into_cat_upper() {
    let domain_1 = get_domain_unit();
    let domain_2 = get_domain_cat_2();

    let point = 1.0;

    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Unit to Cat");
    assert_eq!(
        mapped, "sigmoid",
        "Mapping upper bound of Unit to Cat does not match"
    )
}
#[test]
#[should_panic]
fn unit_intocatt_oob() {
    let domain_1 = get_domain_unit();
    let domain_2 = get_domain_cat_2();

    let point = 1.1;
    let check = "sigmoid";
    
    let mapped = domain_1
        .onto(&point, &domain_2)
        .expect("Error in mapping upper bound from Unit to Cat");
    assert_eq!(
        mapped, check,
        "Mapping upper bound of Unit to Cat does not match"
    )
}
