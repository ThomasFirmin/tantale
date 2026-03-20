use tantale::core::domain::codomain::{
    ElemConstCodomain, ElemConstMultiCodomain, ElemCostCodomain, ElemCostConstCodomain,
    ElemCostConstMultiCodomain, ElemCostMultiCodomain, ElemMultiCodomain, ElemSingleCodomain,
};

#[test]
fn new_elemsinglecodomain() {
    let elem = ElemSingleCodomain::new(1.1);

    assert_eq!(elem.value, 1.1, "Wrong value for ElemSingleCodomain::new.");
}

#[test]
fn new_elemcostcodomain() {
    let elem = ElemCostCodomain::new(1.1, 2.2);

    assert_eq!(elem.value, 1.1, "Wrong value for ElemCostCodomain::new.");
    assert_eq!(elem.cost, 2.2, "Wrong cost for ElemCostCodomain::new.");
}

#[test]
fn new_elemconstcodomain() {
    let elem = ElemConstCodomain::new(1.1, vec![3.3, 4.4]);

    assert_eq!(elem.value, 1.1, "Wrong value for ElemConstCodomain::new.");
    assert_eq!(
        elem.constraints.as_ref(),
        [3.3, 4.4].as_ref(),
        "Wrong constraints for ElemConstCodomain::new."
    );
}

#[test]
fn new_elemcostconstcodomain() {
    let elem = ElemCostConstCodomain::new(1.1, 2.2, vec![3.3, 4.4]);

    assert_eq!(
        elem.value, 1.1,
        "Wrong value for ElemCostConstCodomain::new."
    );
    assert_eq!(elem.cost, 2.2, "Wrong cost for ElemCostConstCodomain::new.");
    assert_eq!(
        elem.constraints.as_ref(),
        [3.3, 4.4].as_ref(),
        "Wrong constraints for ElemCostConstCodomain::new."
    );
}

#[test]
fn new_elemmulticodomain() {
    let elem = ElemMultiCodomain::new(vec![1.1, 2.2]);

    assert_eq!(
        elem.value.as_ref(),
        [1.1, 2.2].as_ref(),
        "Wrong values for ElemMultiCodomain::new."
    );
}

#[test]
fn new_elemcostmulticodomain() {
    let elem = ElemCostMultiCodomain::new(vec![1.1, 2.2], 3.3);

    assert_eq!(
        elem.value.as_ref(),
        [1.1, 2.2].as_ref(),
        "Wrong values for ElemCostMultiCodomain::new."
    );
    assert_eq!(elem.cost, 3.3, "Wrong cost for ElemCostMultiCodomain::new.");
}

#[test]
fn new_elemconstmulticodomain() {
    let elem = ElemConstMultiCodomain::new(vec![1.1, 2.2], vec![3.3, 4.4]);

    assert_eq!(
        elem.value.as_ref(),
        [1.1, 2.2].as_ref(),
        "Wrong values for ElemConstMultiCodomain::new."
    );
    assert_eq!(
        elem.constraints.as_ref(),
        [3.3, 4.4].as_ref(),
        "Wrong constraints for ElemConstMultiCodomain::new."
    );
}

#[test]
fn new_elemcostconstmulticodomain() {
    let elem = ElemCostConstMultiCodomain::new(vec![1.1, 2.2], 3.3, vec![4.4, 5.5]);

    assert_eq!(
        elem.value.as_ref(),
        [1.1, 2.2].as_ref(),
        "Wrong values for ElemCostConstMultiCodomain::new."
    );
    assert_eq!(
        elem.cost, 3.3,
        "Wrong cost for ElemCostConstMultiCodomain::new."
    );
    assert_eq!(
        elem.constraints.as_ref(),
        [4.4, 5.5].as_ref(),
        "Wrong constraints for ElemCostConstMultiCodomain::new."
    );
}
