use tantale::core::{domain::codomain::{
    ElemConstCodomain, ElemConstMultiCodomain, ElemCostCodomain, ElemCostConstCodomain,
    ElemCostConstMultiCodomain, ElemCostMultiCodomain, ElemMultiCodomain, ElemSingleCodomain,
}, utils::xy::YToNdArray};

#[test]
fn yarray_elemsinglecodomain() {
    let elem = ElemSingleCodomain::new(1.1);
    let array = elem.y_array();
    assert_eq!(elem.value, array[[0, 0]], "Mismatch between y_array and value");
}

#[test]
fn yarray_elemcostcodomain() {
    let elem = ElemCostCodomain::new(1.1, 2.2);
    let array = elem.y_array();
    assert_eq!(elem.value, array[[0, 0]], "Mismatch between y_array and value");
}

#[test]
fn yarray_elemconstcodomain() {
    let elem = ElemConstCodomain::new(1.1, vec![3.3, 4.4]);
    let array = elem.y_array();
    assert_eq!(elem.value, array[[0, 0]], "Mismatch between y_array and value");
}

#[test]
fn yarray_elemcostconstcodomain() {
    let elem = ElemCostConstCodomain::new(1.1, 2.2, vec![3.3, 4.4]);
    let array = elem.y_array();
    assert_eq!(elem.value, array[[0, 0]], "Mismatch between y_array and value");
}

#[test]
fn yarray_elemmulticodomain() {
    let elem = ElemMultiCodomain::new(vec![1.1, 2.2]);
    let array = elem.y_array();
    for (i, value) in elem.value.iter().enumerate() {
        assert_eq!(*value, array[[0, i]], "Mismatch between y_array and value");
    }
}

#[test]
fn yarray_elemcostmulticodomain() {
    let elem = ElemCostMultiCodomain::new(vec![1.1, 2.2], 3.3);
    let array = elem.y_array();
    for (i, value) in elem.value.iter().enumerate() {
        assert_eq!(*value, array[[0, i]], "Mismatch between y_array and value");
    }
}

#[test]
fn yarray_elemconstmulticodomain() {
    let elem = ElemConstMultiCodomain::new(vec![1.1, 2.2], vec![3.3, 4.4]);
    let array = elem.y_array();
    for (i, value) in elem.value.iter().enumerate() {
        assert_eq!(*value, array[[0, i]], "Mismatch between y_array and value");
    }
}

#[test]
fn yarray_elemcostconstmulticodomain() {
    let elem = ElemCostConstMultiCodomain::new(vec![1.1, 2.2], 3.3, vec![4.4, 5.5]);
    let array = elem.y_array();
    for (i, value) in elem.value.iter().enumerate() {
        assert_eq!(*value, array[[0, i]], "Mismatch between y_array and value");
    }
}



#[test]
fn yarray_vecelemsinglecodomain() {
    let elem = [
        ElemSingleCodomain::new(1.1),
        ElemSingleCodomain::new(2.2),
        ElemSingleCodomain::new(3.3)
    ];
    let array = elem.y_array();
    for (rowa, rowb) in array.rows().into_iter().zip(elem.iter()) {
        assert_eq!(rowa[[0]], rowb.value, "Mismatch between y_array and value");
    }
}
#[test]
fn yarray_vecelemmulticodomain() {
    let elem = [
        ElemMultiCodomain::new(vec![1.1, 2.2]),
        ElemMultiCodomain::new(vec![2.2, 3.3]),
        ElemMultiCodomain::new(vec![3.3, 4.4]),
    ];
    let array = elem.y_array();
     for (rowa, rowb) in array.rows().into_iter().zip(elem.iter()) {
        for (elema, elemb) in rowa.iter().zip(rowb.value.iter()) {
            assert_eq!(elema, elemb, "Mismatch between y_array and value");
        }
     }
}