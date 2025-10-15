use super::init_cod::*;
use tantale::core::saver::csvsaver::CSVWritable;

#[test]
fn test_elemsingle_header() {
    let (cod, _) = get_elemsingle();
    let head = SingleCodomain::header(&cod);
    let str_true = Vec::from([String::from("y")]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidel_header() {
    let (cod, _) = get_elemfidel();
    let head = CostCodomain::header(&cod);
    let str_true = Vec::from([String::from("y"), String::from("fidelity")]);
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
fn test_elemfidelconst_header() {
    let (cod, _) = get_elemfidelconst();
    let head = CostConstCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y"),
        String::from("fidelity"),
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
fn test_elemfidelmulti_header() {
    let (cod, _) = get_elemfidelmulti();
    let head = CostMultiCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("fidelity"),
    ]);
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
fn test_elemfidelconstmulti_header() {
    let (cod, _) = get_elemfidelconstmulti();
    let head = CostConstMultiCodomain::header(&cod);
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("fidelity"),
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
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidel_write() {
    let (cod, elem) = get_elemfidel();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string()]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemconst_write() {
    let (cod, elem) = get_elemconst();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string(), 3.3.to_string()]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidelconst_write() {
    let (cod, elem) = get_elemfidelconst();
    let head = cod.write(&elem);
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        3.3.to_string(),
        4.4.to_string(),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemmulti_write() {
    let (cod, elem) = get_elemmulti();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string()]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidelmulti_write() {
    let (cod, elem) = get_elemfidelmulti();
    let head = cod.write(&elem);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string(), 3.3.to_string()]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
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
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidelconstmulti_write() {
    let (cod, elem) = get_elemfidelconstmulti();
    let head = cod.write(&elem);
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        3.3.to_string(),
        4.4.to_string(),
        5.5.to_string(),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
