use num::Float;
use tantale::algos::bayesian::bandwidth::{cat_bw, magic_clip, optuna_bw};
use tantale::core::{Bernoulli, Bool, Cat, Unit};
use tantale::core::{Int, Nat, Real, sampler::Uniform};

const EPS: f64 = 1e-10;

// ---- optuna_bw ----

#[test]
fn test_optuna_bw_real() {
    let dom = Real::new(-5.0, 5.0, Uniform);
    let bw = optuna_bw(10, 3, &dom);
    // formula: (up - low)/5 * size^(-1/(dim + 4)) = 1.0/5 * 10^(-1/5)
    let expected = 10.0/5.0 * 10.0.powf(-1.0/7.0);
    assert!(
        (bw - expected).abs() < EPS,
        "optuna_bw Real failed: {} != {}",
        bw,
        expected
    );
}

#[test]
fn test_optuna_bw_int() {
    let dom = Int::new(-5, 5, Uniform);
    let bw = optuna_bw(10, 3, &dom);
    // formula: (up - low)/5 * size^(-1/(dim + 4)) = 1.0/5 * 10^(-1/5)
    let expected = 10.0/5.0 * 10.0.powf(-1.0/7.0);
    assert!(
        (bw - expected).abs() < EPS,
        "optuna_bw Int failed: {} != {}",
        bw,
        expected
    );
}

#[test]
fn test_optuna_bw_nat() {
    let dom = Nat::new(0_u64, 10_u64, Uniform);
    let bw = optuna_bw(10, 3, &dom);
    // formula: (up - low)/5 * size^(-1/(dim + 4)) = 1.0/5 * 10^(-1/5)
    let expected = 10.0/5.0 * 10.0.powf(-1.0/7.0);
    assert!(
        (bw - expected).abs() < EPS,
        "optuna_bw Nat failed: {} != {}",
        bw,
        expected
    );
}

#[test]
fn test_optuna_bw_unit() {
    let dom = Unit::new(Uniform);
    let bw = optuna_bw(10, 3, &dom);
    // formula: (up - low)/5 * size^(-1/(dim + 4)) = 1.0/5 * 10^(-1/5)
    let expected = 1.0/5.0 * 10.0.powf(-1.0/7.0);
    assert!(
        (bw - expected).abs() < EPS,
        "optuna_bw Unit failed: {} != {}",
        bw,
        expected
    );
}

// ---- cat_bw ----

#[test]
fn test_cat_bw_three_categories() {
    let dom = Cat::new(
        ["a", "b", "c"],
        Uniform,
    );
    let bw = cat_bw(10, &dom);
    // formula: (C - 1)/(N + C) = (3 - 1)/(10 + 3) = 2/13
    let expected = 2.0 / 13.0;
    assert!(
        (bw - expected).abs() < EPS,
        "cat_bw 3-cat failed: {} != {}",
        bw,
        expected
    );
}

#[test]
fn test_cat_bw_two_categories() {
    let dom = Cat::new(["yes", "no"], Uniform);
    let bw = cat_bw(8, &dom);
    // formula: (2 - 1)/(8 + 2) = 1/10 = 0.1
    let expected = 1.0 / 10.0;
    assert!(
        (bw - expected).abs() < EPS,
        "cat_bw 2-cat failed: {} != {}",
        bw,
        expected
    );
}

#[test]
fn test_bool_bw_two_categories() {
    let dom = Bool::new(Bernoulli(0.5));
    let bw = cat_bw(8, &dom);
    // formula: (2 - 1)/(8 + 2) = 1/10 = 0.1
    let expected = 1.0 / 10.0;
    assert!(
        (bw - expected).abs() < EPS,
        "cat_bw 2-cat failed: {} != {}",
        bw,
        expected
    );
}

#[test]
fn test_cat_bw_is_always_in_range() {
    // cat_bw should always produce a value in (0, 1)
    let dom = Cat::new(
        ["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()],
        Uniform,
    );
    for n in [1, 5, 10, 100] {
        let bw = cat_bw(n, &dom);
        assert!(bw > 0.0 && bw < 1.0, "cat_bw out of (0,1): {} for n={}", bw, n);
    }
}

// ---- magic_clip ----

#[test]
fn test_magic_clip_clips_small_bandwidth() {
    let dom = Real::new(0.0, 10.0, Uniform);
    // range=10, size=10, min(100, 10)=10, floor = range/10 = 1.0
    let bw_clipped = magic_clip(0.001, 10, &dom);
    assert!(
        (bw_clipped - 1.0).abs() < EPS,
        "magic_clip should clip to 1.0, got: {}",
        bw_clipped
    );
}

#[test]
fn test_magic_clip_no_clip_for_large_bandwidth() {
    let dom = Real::new(0.0, 10.0, Uniform);
    let bw_no_clip = magic_clip(5.0, 10, &dom);
    assert!(
        (bw_no_clip - 5.0).abs() < EPS,
        "magic_clip should not clip 5.0, got: {}",
        bw_no_clip
    );
}

#[test]
fn test_magic_clip_large_size_uses_100_cap() {
    let dom = Real::new(0.0, 100.0, Uniform);
    // range=100, size=200, min(100,200)=100, floor = 100.0/100 = 1.0
    let bw = magic_clip(0.5, 200, &dom);
    assert!(
        (bw - 1.0).abs() < EPS,
        "magic_clip with large size should clip to 1.0, got: {}",
        bw
    );
}
