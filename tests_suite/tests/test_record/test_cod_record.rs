use super::init_cod::*;
use tantale::core::utils::record::{Schema, Recordable};

#[test]
fn test_elemsingle_schema() {
    let (cod, _) = get_elemsingle();
    let head = ElemSingleCodomain::columns(&cod, ());
    let str_true = Vec::from([String::from("y")]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemcost_schema() {
    let (cod, _) = get_elemcost();
    let head = ElemCostCodomain::columns(&cod, ());
    let str_true = Vec::from([String::from("y"), String::from("cost")]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemconst_schema() {
    let (cod, _) = get_elemconst();
    let head = ElemConstCodomain::columns(&cod, ());
    let str_true = Vec::from([
        String::from("y"),
        String::from("c0"),
        String::from("c1"),
        String::from("c2"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemcostconst_schema() {
    let (cod, _) = get_elemcostconst();
    let head = ElemCostConstCodomain::columns(&cod, ());
    let str_true = Vec::from([
        String::from("y"),
        String::from("cost"),
        String::from("c0"),
        String::from("c1"),
        String::from("c2"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemmulti_schema() {
    let (cod, _) = get_elemmulti();
    let head = ElemMultiCodomain::columns(&cod, ());
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("y2"),
        String::from("y3"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemcostmulti_schema() {
    let (cod, _) = get_elemcostmulti();
    let head = ElemCostMultiCodomain::columns(&cod, ());
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("y2"),
        String::from("y3"),
        String::from("cost"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemconstmulti_schema() {
    let (cod, _) = get_elemconstmulti();
    let head = ElemConstMultiCodomain::columns(&cod, ());
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("y2"),
        String::from("y3"),
        String::from("c0"),
        String::from("c1"),
        String::from("c2"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemcostconstmulti_schema() {
    let (cod, _) = get_elemcostconstmulti();
    let head = ElemCostConstMultiCodomain::columns(&cod, ());
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("y2"),
        String::from("y3"),
        String::from("cost"),
        String::from("c0"),
        String::from("c1"),
        String::from("c2"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}

#[test]
fn test_elemsingle_write() {
    let (cod, elem) = get_elemsingle();
    let head = elem.record(&cod, ());
    let str_true = Vec::from([1.1.to_string()]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
#[test]
fn test_elemcost_write() {
    let (cod, elem) = get_elemcost();
    let head = elem.record(&cod, ());
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string()]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
#[test]
fn test_elemconst_write() {
    let (cod, elem) = get_elemconst();
    let head = elem.record(&cod, ());
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        3.3.to_string(),
        4.4.to_string(),
    ]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
#[test]
fn test_elemcostconst_write() {
    let (cod, elem) = get_elemcostconst();
    let head = elem.record(&cod, ());
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        3.3.to_string(),
        4.4.to_string(),
    ]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
#[test]
fn test_elemmulti_write() {
    let (cod, elem) = get_elemmulti();
    let head = elem.record(&cod, ());
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        3.3.to_string(),
        4.4.to_string(),
    ]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
#[test]
fn test_elemcostmulti_write() {
    let (cod, elem) = get_elemcostmulti();
    let head = elem.record(&cod, ());
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        3.3.to_string(),
        4.4.to_string(),
        3.3.to_string(),
    ]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
#[test]
fn test_elemconstmulti_write() {
    let (cod, elem) = get_elemconstmulti();
    let head = elem.record(&cod, ());
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        3.3.to_string(),
        4.4.to_string(),
        3.3.to_string(),
        4.4.to_string(),
    ]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
#[test]
fn test_elemcostconstmulti_write() {
    let (cod, elem) = get_elemcostconstmulti();
    let head = elem.record(&cod, ());
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        3.3.to_string(),
        4.4.to_string(),
        3.3.to_string(),
        4.4.to_string(),
        5.5.to_string(),
    ]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
