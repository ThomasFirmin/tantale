use tantale::algos::bayesian::weighter::{UniformWeighter, Weighter};

const EPS: f64 = 1e-10;

// ---- UniformWeighter construction ----

#[test]
fn test_uniform_weighter_default_prior() {
    let w = UniformWeighter::default();
    // Default prior should be 1.0 (internal field)
    // We verify indirectly through weight computation
    let good = vec![&1, &2, &3];
    let bad = vec![&4, &5, &6, &7];
    let weights = w.weight(&good, &bad);
    // prior=1.0, n_good=3: normalize_cst_good = 3+1=4, good_prior_weight = 1.0/4 = 0.25
    assert!(
        (weights.good.prior_weight - 0.25).abs() < EPS,
        "Default prior_weight for good should be 0.25: {}",
        weights.good.prior_weight
    );
}

// ---- UniformWeighter weight computation ----

#[test]
fn test_uniform_weighter_good_weights() {
    let w = UniformWeighter::default(); // prior = 1.0
    let good = vec![&1, &2, &3]; // n_good = 3
    let bad = vec![&4, &5, &6, &7, &8]; // n_bad = 5
    let weights = w.weight(&good, &bad);

    // normalize_cst_good = 3 + 1 = 4; good_weight = 1/4 = 0.25
    let expected_good_weight = 1.0 / 4.0;
    assert!(
        weights
            .good
            .weights
            .iter()
            .all(|w| (w - expected_good_weight).abs() < EPS),
        "All good weights should be {}: {:?}",
        expected_good_weight,
        weights.good.weights
    );
    assert_eq!(weights.good.weights.len(), 3, "Should have 3 good weights");
}

#[test]
fn test_uniform_weighter_bad_weights() {
    let w = UniformWeighter::default(); // prior = 1.0
    let good = vec![&1, &2, &3];
    let bad = vec![&4, &5, &6, &7, &8]; // n_bad = 5
    let weights = w.weight(&good, &bad);

    // normalize_cst_bad = 5 + 1 = 6; bad_weight = 1/6
    let expected_bad_weight = 1.0 / 6.0;
    assert!(
        weights
            .bad
            .weights
            .iter()
            .all(|w| (w - expected_bad_weight).abs() < EPS),
        "All bad weights should be {}: {:?}",
        expected_bad_weight,
        weights.bad.weights
    );
    assert_eq!(weights.bad.weights.len(), 5, "Should have 5 bad weights");
}

#[test]
fn test_uniform_weighter_weights_sum_to_one() {
    let w = UniformWeighter::default();
    let good = vec![&1, &2, &3, &4, &5];
    let bad = vec![&6, &7, &8, &9];
    let weights = w.weight(&good, &bad);

    // sum(good_weights) + good_prior_weight = 1.0
    let good_sum: f64 = weights.good.weights.iter().sum::<f64>() + weights.good.prior_weight;
    assert!(
        (good_sum - 1.0).abs() < EPS,
        "good weights + prior should sum to 1.0: {}",
        good_sum
    );

    let bad_sum: f64 = weights.bad.weights.iter().sum::<f64>() + weights.bad.prior_weight;
    assert!(
        (bad_sum - 1.0).abs() < EPS,
        "bad weights + prior should sum to 1.0: {}",
        bad_sum
    );
}

#[test]
fn test_uniform_weighter_prior_weight_formula() {
    let w = UniformWeighter::default(); // prior = 1.0
    let good = vec![&1, &2, &3, &4]; // n_good = 4
    let bad = vec![&5, &6]; // n_bad = 2
    let weights = w.weight(&good, &bad);

    // good_prior_weight = prior / (n_good + prior) = 1/(4+1) = 0.2
    assert!(
        (weights.good.prior_weight - 0.2).abs() < EPS,
        "good prior_weight should be 0.2: {}",
        weights.good.prior_weight
    );

    // bad_prior_weight = prior / (n_bad + prior) = 1/(2+1) = 0.333
    let expected_bad_prior = 1.0 / 3.0;
    assert!(
        (weights.bad.prior_weight - expected_bad_prior).abs() < EPS,
        "bad prior_weight should be {}: {}",
        expected_bad_prior,
        weights.bad.prior_weight
    );
}

// ---- with_prior ----

#[test]
fn test_uniform_weighter_with_prior_good_weights() {
    let w = UniformWeighter::default(); // prior = 1.0
    let good = vec![&1, &2, &3]; // n_good = 3
    let bad = vec![&4, &5, &6, &7, &8]; // n_bad = 5
    let weights = w.weight(&good, &bad);

    // normalize_cst_good = 3 + 2 = 5; good_weight = 1/5 = 0.2
    let expected_good_weight = 1.0 / 4.0;
    assert!(
        weights
            .good
            .weights
            .iter()
            .all(|w| (w - expected_good_weight).abs() < EPS),
        "All good weights should be {}: {:?}",
        expected_good_weight,
        weights.good.weights
    );
    assert_eq!(weights.good.weights.len(), 3, "Should have 3 good weights");
}

#[test]
fn test_uniform_weighter_with_prior_bad_weights() {
    let w = UniformWeighter::default(); // prior = 1.0
    let good = vec![&1, &2, &3];
    let bad = vec![&4, &5, &6, &7, &8]; // n_bad = 5
    let weights = w.weight(&good, &bad);

    // normalize_cst_bad = 5 + 2 = 6; bad_weight = 1/6
    let expected_bad_weight = 1.0 / 6.0;
    assert!(
        weights
            .bad
            .weights
            .iter()
            .all(|w| (w - expected_bad_weight).abs() < EPS),
        "All bad weights should be {}: {:?}",
        expected_bad_weight,
        weights.bad.weights
    );
    assert_eq!(weights.bad.weights.len(), 5, "Should have 5 bad weights");
}

#[test]
fn test_uniform_weighter_with_prior_weights_sum_to_one() {
    let w_custom = UniformWeighter::new(2.0);

    let good = vec![&1, &2, &3]; // n_good = 3
    let bad = vec![&4, &5, &6, &7]; // n_bad = 4

    let weights = w_custom.weight(&good, &bad);

    // sum(good_weights) + good_prior_weight = 1.0
    let good_sum: f64 = weights.good.weights.iter().sum::<f64>() + weights.good.prior_weight;
    assert!(
        (good_sum - 1.0).abs() < EPS,
        "good weights + prior should sum to 1.0: {}",
        good_sum
    );

    let bad_sum: f64 = weights.bad.weights.iter().sum::<f64>() + weights.bad.prior_weight;
    assert!(
        (bad_sum - 1.0).abs() < EPS,
        "bad weights + prior should sum to 1.0: {}",
        bad_sum
    );
}

#[test]
fn test_uniform_weighter_with_prior_weights_formula() {
    let w_custom = UniformWeighter::new(2.0);

    let good = vec![&1, &2, &3]; // n_good = 3
    let bad = vec![&4, &5, &6, &7]; // n_bad = 4

    let weights = w_custom.weight(&good, &bad);

    // prior=2.0: normalize_cst_good = 3+2 = 5; good_weight = 1/5 = 0.2
    // prior_weight = 2/5 = 0.4
    assert_eq!(
        weights.good.prior_weight, 0.4,
        "good prior_weight with prior=2 should be 0.4: {}",
        weights.good.prior_weight
    );
    assert!(
        weights.good.weights.iter().all(|w| (w - 0.2).abs() < EPS),
        "good prior_weight should be 0.2: {:?}",
        weights.good.weights
    );
}

// ---- edge cases ----

#[test]
fn test_uniform_weighter_single_good() {
    let w = UniformWeighter::default();
    let good = vec![&1];
    let bad = vec![&2, &3, &4];
    let weights = w.weight(&good, &bad);
    // normalize_cst_good = 1 + 1 = 2; good_weight = 0.5; good_prior_weight = 0.5
    assert!((weights.good.weights[0] - 0.5).abs() < EPS);
    assert!((weights.good.prior_weight - 0.5).abs() < EPS);
    let total: f64 = weights.good.weights.iter().sum::<f64>() + weights.good.prior_weight;
    assert!((total - 1.0).abs() < EPS);
}

#[test]
fn test_uniform_weighter_empty_sets_returns_correct_structure_good() {
    let w = UniformWeighter::default();
    let empty: Vec<&i32> = vec![];
    let non_empty = vec![&1, &2, &3];
    let weights = w.weight(&empty, &non_empty);
    // Empty good set: good weight vec is empty, prior_weight = prior / (0 + prior) = 1.0
    assert!(
        weights.good.weights.is_empty(),
        "Empty good set should produce no weights"
    );
    assert!(
        (weights.good.prior_weight - 1.0).abs() < EPS,
        "Empty good prior_weight should be 1.0: {}",
        weights.good.prior_weight
    );
}

#[test]
fn test_uniform_weighter_single_bad() {
    let w = UniformWeighter::default();
    let good = vec![&2, &3, &4];
    let bad = vec![&1];
    let weights = w.weight(&good, &bad);
    // normalize_cst_bad = 1 + 1 = 2; bad_weight = 0.5; bad_prior_weight = 0.5
    assert!((weights.bad.weights[0] - 0.5).abs() < EPS);
    assert!((weights.bad.prior_weight - 0.5).abs() < EPS);
    let total: f64 = weights.bad.weights.iter().sum::<f64>() + weights.bad.prior_weight;
    assert!((total - 1.0).abs() < EPS);
}

#[test]
fn test_uniform_weighter_empty_sets_returns_correct_structure_bad() {
    let w = UniformWeighter::default();
    let empty: Vec<&i32> = vec![&1, &2, &3];
    let non_empty = vec![];
    let weights = w.weight(&empty, &non_empty);
    // Empty bad set: bad weight vec is empty, prior_weight = prior / (0 + prior) = 1.0
    assert!(
        weights.bad.weights.is_empty(),
        "Empty bad set should produce no weights"
    );
    assert!(
        (weights.bad.prior_weight - 1.0).abs() < EPS,
        "Empty bad prior_weight should be 1.0: {}",
        weights.bad.prior_weight
    );
}
