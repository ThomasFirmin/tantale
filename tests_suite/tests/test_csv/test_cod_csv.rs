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
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}
#[test]
fn test_elemcost_write() {
    let (cod, elem) = get_elemcost();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string()]);
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}
#[test]
fn test_elemconst_write() {
    let (cod, elem) = get_elemconst();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string(), 3.3.to_string()]);
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
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
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}
#[test]
fn test_elemmulti_write() {
    let (cod, elem) = get_elemmulti();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string()]);
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}
#[test]
fn test_elemcostmulti_write() {
    let (cod, elem) = get_elemcostmulti();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string(), 3.3.to_string()]);
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
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
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
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
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}

#[test]
fn test_elemfidfid_header() {
    let (cod, _) = get_elemfid();
    let head = StepCodomain::header(&cod);
    let str_true = Vec::from([String::from("y"), String::from("step")]);
    assert_eq!(head, str_true, "Written linedoes not match the true baseline.");
}
#[test]
fn test_elemfidcost_header() {
    let (cod, _) = get_elemfidcost();
    let head = StepCostCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y"),
        String::from("step"),
        String::from("cost"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidconst_header() {
    let (cod, _) = get_elemfidconst();
    let head = StepConstCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y"),
        String::from("step"),
        String::from("c0"),
        String::from("c1"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidcostconst_header() {
    let (cod, _) = get_elemfidcostconst();
    let head = StepCostConstCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y"),
        String::from("step"),
        String::from("cost"),
        String::from("c0"),
        String::from("c1"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidmulti_header() {
    let (cod, _) = get_elemfidmulti();
    let head = StepMultiCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("step"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidcostmulti_header() {
    let (cod, _) = get_elemfidcostmulti();
    let head = StepCostMultiCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("step"),
        String::from("cost"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidconstmulti_header() {
    let (cod, _) = get_elemfidconstmulti();
    let head = StepConstMultiCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("step"),
        String::from("c0"),
        String::from("c1"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidcostconstmulti_header() {
    let (cod, _) = get_elemfidcostconstmulti();
    let head = StepCostConstMultiCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("step"),
        String::from("cost"),
        String::from("c0"),
        String::from("c1"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}

#[test]
fn test_elemfid_write() {
    let (cod, elem) = get_elemfid();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), "Completed".to_string()]);
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}
#[test]
fn test_elemfidcost_write() {
    let (cod, elem) = get_elemfidcost();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), "Completed".to_string(), 2.2.to_string()]);
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}
#[test]
fn test_elemfidconst_write() {
    let (cod, elem) = get_elemfidconst();
    let head = cod.write(&elem);
    let str_true = Vec::from([
        1.1.to_string(),
        "Completed".to_string(),
        2.2.to_string(),
        3.3.to_string(),
    ]);
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}
#[test]
fn test_elemfidcostconst_write() {
    let (cod, elem) = get_elemfidcostconst();
    let head = cod.write(&elem);
    let str_true = Vec::from([
        1.1.to_string(),
        "Completed".to_string(),
        2.2.to_string(),
        3.3.to_string(),
        4.4.to_string(),
    ]);
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}
#[test]
fn test_elemfidmulti_write() {
    let (cod, elem) = get_elemfidmulti();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string(), "Completed".to_string()]);
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}
#[test]
fn test_elemfidcostmulti_write() {
    let (cod, elem) = get_elemfidcostmulti();
    let head = cod.write(&elem);
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        "Completed".to_string(),
        3.3.to_string(),
    ]);
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}
#[test]
fn test_elemfidconstmulti_write() {
    let (cod, elem) = get_elemfidconstmulti();
    let head = cod.write(&elem);
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        "Completed".to_string(),
        3.3.to_string(),
        4.4.to_string(),
    ]);
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}
#[test]
fn test_elemfidcostconstmulti_write() {
    let (cod, elem) = get_elemfidcostconstmulti();
    let head = cod.write(&elem);
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        "Completed".to_string(),
        3.3.to_string(),
        4.4.to_string(),
        5.5.to_string(),
    ]);
    assert_eq!(head, str_true, "Written line does not match the true baseline.");
}
