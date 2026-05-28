use tantale::algos::bayesian::splitter::{LinearSplit, Splitter, SqrtSplit};
use tantale::algos::utils::OrdArchive;

fn make_archive<I: IntoIterator<Item = i32>>(values: I) -> OrdArchive<i32> {
    let mut archive = OrdArchive::default();
    for v in values {
        archive.add(v);
    }
    archive
}

// ---- LinearSplit construction ----

#[test]
fn test_linear_split_valid() {
    let result = LinearSplit::new(0.25);
    assert!(result.is_ok(), "LinearSplit(0.25) should be valid");
}

#[test]
fn test_linear_split_invalid_zero() {
    let result = LinearSplit::new(0.0);
    assert!(
        result.is_err(),
        "LinearSplit(0.0) should fail (beta must be > 0)"
    );
}

#[test]
fn test_linear_split_invalid_one() {
    let result = LinearSplit::new(1.0);
    assert!(
        result.is_err(),
        "LinearSplit(1.0) should fail (beta must be < 1)"
    );
}

#[test]
fn test_linear_split_invalid_negative() {
    let result = LinearSplit::new(-0.1);
    assert!(result.is_err(), "LinearSplit(-0.1) should fail");
}

#[test]
fn test_linear_split_invalid_greater_one() {
    let result = LinearSplit::new(1.5);
    assert!(result.is_err(), "LinearSplit(1.5) should fail");
}

// ---- LinearSplit splitting ----

#[test]
fn test_linear_split_sizes() {
    let archive = make_archive(0..10);
    let splitter = LinearSplit::new(0.25).unwrap();
    let (top, bottom) = splitter.split(&archive);
    // quantile = floor(0.25 * 10) = 2
    // top = points[2..] => 2 elements; bottom = points[..2] => 8 elements
    assert_eq!(top.len(), 2, "top slice should have 2 elements");
    assert_eq!(bottom.len(), 8, "bottom slice should have 8 elements");
}

#[test]
fn test_linear_split_half() {
    let archive = make_archive(0..10);
    let splitter = LinearSplit::new(0.5).unwrap();
    let (top, bottom) = splitter.split(&archive);
    // quantile = floor(0.5 * 10) = 5
    assert_eq!(top.len(), 5);
    assert_eq!(bottom.len(), 5);
}

#[test]
fn test_linear_split_covers_all_elements() {
    let archive = make_archive(0..10);
    let splitter = LinearSplit::new(0.3).unwrap();
    let (top, bottom) = splitter.split(&archive);
    assert_eq!(
        top.len() + bottom.len(),
        archive.size(),
        "top + bottom should cover all elements"
    );
}

#[test]
fn test_linear_split_values_are_sorted() {
    let archive = make_archive([5, 2, 8, 1, 9, 3]);
    let splitter = LinearSplit::new(0.5).unwrap();
    let (top, bottom) = splitter.split(&archive);
    // Archive is stored in sorted ascending order: [1, 2, 3, 5, 8, 9]
    // quantile = 3; top = [5, 8, 9], bottom = [1, 2, 3]
    assert_eq!(top, &[5, 8, 9]);
    assert_eq!(bottom, &[1, 2, 3]);
}

#[test]
fn test_linear_split_small_archive() {
    // With 3 elements and beta=0.33, quantile=0, so top=all, bottom=none
    let archive = make_archive([1, 2, 3]);
    let splitter = LinearSplit::new(0.25).unwrap();
    let (top, bottom) = splitter.split(&archive);
    // quantile = floor(0.25 * 3) = 0
    assert_eq!(top.len() + bottom.len(), 3);
}

#[test]
fn test_linear_split_one_archive() {
    let archive = make_archive([42]);
    let splitter = LinearSplit::new(0.25).unwrap();
    let (top, bottom) = splitter.split(&archive);
    // quantile = floor(0.25 * 1) = 0; top=all, bottom=none
    assert_eq!(top.len(), 0);
    assert_eq!(bottom.len(), 1);
}

// ---- SqrtSplit construction ----

#[test]
fn test_sqrt_split_valid() {
    let result = SqrtSplit::new(1.0);
    assert!(result.is_ok(), "SqrtSplit(1.0) should be valid");
}

#[test]
fn test_sqrt_split_invalid_zero() {
    let result = SqrtSplit::new(0.0);
    assert!(
        result.is_err(),
        "SqrtSplit(0.0) should fail (beta must be > 0)"
    );
}

#[test]
fn test_sqrt_split_invalid_negative() {
    let result = SqrtSplit::new(-1.0);
    assert!(result.is_err(), "SqrtSplit(-1.0) should fail");
}

// ---- SqrtSplit splitting ----

#[test]
fn test_sqrt_split_sizes() {
    let archive = make_archive(0..9);
    let splitter = SqrtSplit::new(4.0).unwrap();
    let (top, bottom) = splitter.split(&archive);
    // quantile = floor(1.0 * sqrt(9)) = floor(3.0) = 3
    // top = points[3..] = 6 elements; bottom = points[..3] = 3 elements
    assert_eq!(top.len(), 1, "top slice should have 1 elements");
    assert_eq!(bottom.len(), 8, "bottom slice should have 8 elements");
}

#[test]
fn test_sqrt_split_covers_all_elements() {
    let archive = make_archive(0..16);
    let splitter = SqrtSplit::new(4.0).unwrap();
    let (top, bottom) = splitter.split(&archive);
    assert_eq!(
        top.len() + bottom.len(),
        archive.size(),
        "top + bottom should cover all elements"
    );
}

#[test]
fn test_sqrt_split_values_are_sorted() {
    let archive = make_archive(vec![7_i32, 2, 8, 1, 9, 3, 6, 0, 4, 5]);
    let splitter = SqrtSplit::new(4.0).unwrap();
    let (top, bottom) = splitter.split(&archive);
    assert_eq!(top, &[9]);
    assert_eq!(bottom, &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn test_sqrt_split_small_archive() {
    let archive = make_archive([1, 2, 3]);
    let splitter = SqrtSplit::new(4.0).unwrap();
    let (top, bottom) = splitter.split(&archive);
    assert_eq!(top.len() + bottom.len(), 3);
}

#[test]
fn test_sqrt_split_one_archive() {
    let archive = make_archive([42]);
    let splitter = SqrtSplit::new(4.0).unwrap();
    let (top, bottom) = splitter.split(&archive);
    assert_eq!(top.len(), 0);
    assert_eq!(bottom.len(), 1);
}
