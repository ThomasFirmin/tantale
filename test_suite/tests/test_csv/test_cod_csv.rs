use super::init_cod::*;
use tantale::core::saver::csvsaver::CSVWritable;

#[test]
fn test_elemsingle_header() {
    let cod = get_elemsingle();
    let head = cod.header();
    let str_true = Vec::from([String::from("y")]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidel_header() {
    let cod = get_elemfidel();
    let head = cod.header();
    let str_true = Vec::from([String::from("y"), String::from("fidelity")]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemconst_header() {
    let cod = get_elemconst();
    let head = cod.header();
    let str_true = Vec::from([String::from("y"), String::from("c0"), String::from("c1")]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidelconst_header() {
    let cod = get_elemfidelconst();
    let head = cod.header();
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
    let cod = get_elemmulti();
    let head = cod.header();
    let str_true = Vec::from([String::from("y0"), String::from("y1")]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidelmulti_header() {
    let cod = get_elemfidelmulti();
    let head = cod.header();
    let str_true = Vec::from([
        String::from("y0"),
        String::from("y1"),
        String::from("fidelity"),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemconstmulti_header() {
    let cod = get_elemconstmulti();
    let head = cod.header();
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
    let cod = get_elemfidelconstmulti();
    let head = cod.header();
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
    let cod = get_elemsingle();
    let head = cod.write(&cod);
    let str_true = Vec::from([1.1.to_string()]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidel_write() {
    let cod = get_elemfidel();
    let head = cod.write(&cod);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string()]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemconst_write() {
    let cod = get_elemconst();
    let head = cod.write(&cod);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string(), 3.3.to_string()]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidelconst_write() {
    let cod = get_elemfidelconst();
    let head = cod.write(&cod);
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
    let cod = get_elemmulti();
    let head = cod.write(&cod);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string()]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemfidelmulti_write() {
    let cod = get_elemfidelmulti();
    let head = cod.write(&cod);
    let str_true = Vec::from([1.1.to_string(), 2.2.to_string(), 3.3.to_string()]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
#[test]
fn test_elemconstmulti_write() {
    let cod = get_elemconstmulti();
    let head = cod.write(&cod);
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
    let cod = get_elemfidelconstmulti();
    let head = cod.write(&cod);
    let str_true = Vec::from([
        1.1.to_string(),
        2.2.to_string(),
        3.3.to_string(),
        4.4.to_string(),
        5.5.to_string(),
    ]);
    assert_eq!(head, str_true, "Header does not match the true baseline.");
}
