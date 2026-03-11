use super::init_cod::*;
use tantale::core::recorder::csv::CSVWritable;

#[test]
fn test_elemsingle_header() {
    let (cod, _) = get_elemsingle();
    let head = SingleCodomain::header(&cod);
    let str_true = Vec::from([String::from("y")]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemcost_header() {
    let (cod, _) = get_elemcost();
    let head = CostCodomain::header(&cod);
    let str_true = Vec::from([String::from("y"), String::from("cost")]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemconst_header() {
    let (cod, _) = get_elemconst();
    let head = ConstCodomain::header(&cod);
    let str_true = Vec::from([String::from("y"), String::from("c0"), String::from("c1")]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemcostconst_header() {
    let (cod, _) = get_elemcostconst();
    let head = CostConstCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y"),
        String::from("cost"),
        String::from("c0"),
        String::from("c1"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemmulti_header() {
    let (cod, _) = get_elemmulti();
    let head = MultiCodomain::header(&cod);
    let str_true = Vec::from([String::from("y0"), String::from("y1")]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemcostmulti_header() {
    let (cod, _) = get_elemcostmulti();
    let head = CostMultiCodomain::header(&cod);
    let str_true = Vec::from([String::from("y0"), String::from("y1"), String::from("cost")]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemconstmulti_header() {
    let (cod, _) = get_elemconstmulti();
    let head = ConstMultiCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("c0"),
        String::from("c1"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemcostconstmulti_header() {
    let (cod, _) = get_elemcostconstmulti();
    let head = CostConstMultiCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("cost"),
        String::from("c0"),
        String::from("c1"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}

#[test]
fn test_elemsingle_write() {
    let (cod, elem) = get_elemsingle();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string()]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
#[test]
fn test_elemcost_write() {
    let (cod, elem) = get_elemcost();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string()]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
#[test]
fn test_elemconst_write() {
    let (cod, elem) = get_elemconst();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string(), 3.3.to_string()]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
#[test]
fn test_elemcostconst_write() {
    let (cod, elem) = get_elemcostconst();
    let head = cod.write(&elem);
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
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string()]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
#[test]
fn test_elemcostmulti_write() {
    let (cod, elem) = get_elemcostmulti();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string(), 3.3.to_string()]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
#[test]
fn test_elemconstmulti_write() {
    let (cod, elem) = get_elemconstmulti();
    let head = cod.write(&elem);
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
fn test_elemcostconstmulti_write() {
    let (cod, elem) = get_elemcostconstmulti();
    let head = cod.write(&elem);
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        3.3.to_string(),
        4.4.to_string(),
        5.5.to_string(),
    ]);
    assert_eq!(
        head, str_true,
        "Written line does not match the true baseline."
    );
}
