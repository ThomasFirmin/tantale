use std::f64::consts::PI;

use tantale::algos::bayesian::kernel::{AitchisonAitkenKernel, GaussianKernel, KernelFunc};
use tantale::algos::bayesian::tpe::acquisition;
use tantale::core::{Cat, Domain};
use tantale::core::{Int, Nat, Real, Unit, sampler::Uniform};

const EPS: f64 = 1e-9;

// ---- GaussianKernel on Real ----

#[test]
fn test_gaussian_kernel_real() {
    let dom = Real::new(0.0, 1.0, Uniform);

    let sctx = GaussianKernel::get_scontext(1.0, &dom);
    let kctx = GaussianKernel::get_kcontext(&0.4, &sctx, &dom);

    let k = GaussianKernel::compute(&0.5, &0.4, &kctx, &sctx, &dom);
    let g =
        1.0 / (2.0 * PI * 1.0_f64.powi(2)).sqrt() * (-0.5 * ((0.5 - 0.4) / 1.0_f64).powi(2)).exp();
    let norm_cst = 0.38116862386025063;
    let expected = g / norm_cst;
    assert!(
        (k - expected).abs() < EPS,
        "Gaussian kernel on Real failed: {} != {}",
        k,
        expected
    );
}

#[test]
fn test_gaussian_kernel_real_sample_in_bounds() {
    let dom = Real::new(10.0, 20.0, Uniform);
    let mut rng = rand::rng();

    let sctx = GaussianKernel::get_scontext(5.0, &dom);
    let kctx = GaussianKernel::get_kcontext(&15.0, &sctx, &dom);

    for _ in 0..20 {
        let sample = GaussianKernel::sample(&mut rng, &15.0, &kctx, &sctx, &dom);
        assert!(
            dom.contains(&sample),
            "Sample from Real kernel must be in [10,20]: {}",
            sample
        );
    }
}

#[test]
fn test_gaussian_kernel_real_prior_positive() {
    let dom = Real::new(0.0, 1.0, Uniform);
    let p = GaussianKernel::prior(&0.5, &dom);
    let expected = 0.3989422804014327;
    assert!(
        (p - expected).abs() < EPS,
        "Gaussian kernel prior on Real failed: {} != {}",
        p,
        expected
    );
}

// ---- GaussianKernel on Unit ----

#[test]
fn test_gaussian_kernel_unit_positive() {
    let dom = Unit::new(Uniform);

    let sctx = GaussianKernel::get_scontext(1.0, &dom);
    let kctx = GaussianKernel::get_kcontext(&0.4, &sctx, &dom);

    let k = GaussianKernel::compute(&0.5, &0.4, &kctx, &sctx, &dom);

    let g =
        1.0 / (2.0 * PI * 1.0_f64.powi(2)).sqrt() * (-0.5 * ((0.5 - 0.4) / 1.0_f64).powi(2)).exp();
    let norm_cst = 0.38116862386025063;
    let expected = g / norm_cst;
    assert!(
        (k - expected).abs() < EPS,
        "Gaussian kernel on Unit failed: {} != {}",
        k,
        expected
    );
}

#[test]
fn test_gaussian_kernel_unit_sample_in_bounds() {
    let dom = Unit::new(Uniform);
    let mut rng = rand::rng();
    let sctx = GaussianKernel::get_scontext(0.9, &dom);
    let kctx = GaussianKernel::get_kcontext(&0.4, &sctx, &dom);

    for _ in 0..20 {
        let sample = GaussianKernel::sample(&mut rng, &0.4, &kctx, &sctx, &dom);
        assert!(
            dom.contains(&sample),
            "Sample from Unit kernel must be in [0,1]: {}",
            sample
        );
    }
}

// ---- GaussianKernel on Int ----

#[test]
fn test_gaussian_kernel_int_positive() {
    let dom = Int::new(0, 10, Uniform);

    let sctx = GaussianKernel::get_scontext(5.0, &dom);
    let kctx = GaussianKernel::get_kcontext(&4, &sctx, &dom);

    let k = GaussianKernel::compute(&5, &4, &kctx, &sctx, &dom);
    let g = 0.07808358491192358;
    let norm_cst = 0.7191393900676302;
    let expected = g / norm_cst;
    assert!(
        (k - expected).abs() < EPS,
        "Gaussian kernel on Int failed: {} != {}",
        k,
        expected
    );
}

#[test]
fn test_gaussian_kernel_int_same_point_higher() {
    // Kernel at the same point should be >= kernel at a distant point
    let dom = Int::new(0, 10, Uniform);

    let sctx = GaussianKernel::get_scontext(5.0, &dom);
    let kctx = GaussianKernel::get_kcontext(&4, &sctx, &dom);

    let k_same = GaussianKernel::compute(&4, &4, &kctx, &sctx, &dom);

    let sctx = GaussianKernel::get_scontext(5.0, &dom);
    let kctx = GaussianKernel::get_kcontext(&4, &sctx, &dom);

    let k_far = GaussianKernel::compute(&0, &4, &kctx, &sctx, &dom);
    assert!(
        k_same >= k_far,
        "Kernel at same point k(x1,x1) should be >= distant point k(x1,x2): {} < {}",
        k_same,
        k_far
    );
}

#[test]
fn test_gaussian_kernel_int_sample_in_bounds() {
    let dom = Int::new(0, 10, Uniform);
    let mut rng = rand::rng();

    let sctx = GaussianKernel::get_scontext(5.0, &dom);
    let kctx = GaussianKernel::get_kcontext(&4, &sctx, &dom);

    for _ in 0..20 {
        let sample = GaussianKernel::sample(&mut rng, &5, &kctx, &sctx, &dom);
        assert!(
            dom.contains(&sample),
            "Sample from Int kernel must be in [0,10]: {}",
            sample
        );
    }
}

// ---- GaussianKernel on Nat ----

#[test]
fn test_gaussian_kernel_nat_positive() {
    let dom = Nat::new(0_u64, 10_u64, Uniform);

    let sctx = GaussianKernel::get_scontext(5.0, &dom);
    let kctx = GaussianKernel::get_kcontext(&4, &sctx, &dom);

    let k = GaussianKernel::compute(&5, &4, &kctx, &sctx, &dom);

    let g = 0.07808358491192358;
    let norm_cst = 0.7191393900676302;
    let expected = g / norm_cst;
    assert!(
        (k - expected).abs() < EPS,
        "Gaussian kernel on Nat failed: {} != {}",
        k,
        expected
    );
}

#[test]
fn test_gaussian_kernel_nat_same_point_higher() {
    // Kernel at the same point should be >= kernel at a distant point
    let dom = Nat::new(0_u64, 10_u64, Uniform);
    let sctx = GaussianKernel::get_scontext(5.0, &dom);
    let kctx = GaussianKernel::get_kcontext(&4, &sctx, &dom);

    let k_same = GaussianKernel::compute(&4, &4, &kctx, &sctx, &dom);

    let sctx = GaussianKernel::get_scontext(5.0, &dom);
    let kctx = GaussianKernel::get_kcontext(&4, &sctx, &dom);

    let k_far = GaussianKernel::compute(&0, &4, &kctx, &sctx, &dom);
    assert!(
        k_same >= k_far,
        "Kernel at same point k(x1,x1) should be >= distant point k(x1,x2): {} < {}",
        k_same,
        k_far
    );
}

#[test]
fn test_gaussian_kernel_nat_sample_in_bounds() {
    let dom = Nat::new(0_u64, 10_u64, Uniform);
    let mut rng = rand::rng();

    let sctx = GaussianKernel::get_scontext(5.0, &dom);
    let kctx = GaussianKernel::get_kcontext(&4, &sctx, &dom);

    for _ in 0..20 {
        let sample = GaussianKernel::sample(&mut rng, &5, &kctx, &sctx, &dom);
        assert!(
            dom.contains(&sample),
            "Sample from Nat kernel must be in [0,10]: {}",
            sample
        );
    }
}

// ---- AitchisonAitkenKernel ----

#[test]
fn test_aitchison_aitken_same_category() {
    let dom = Cat::new(["a", "b", "c"], Uniform);

    let sctx = AitchisonAitkenKernel::get_scontext(0.1, &dom);

    let k = AitchisonAitkenKernel::compute(&"a".to_string(), &"a".to_string(), &(), &sctx, &dom);

    // Formula: 1 - bw = 1 - 0.1 = 0.9
    assert!(
        (k - (1.0 - 0.1)).abs() < EPS,
        "Same-category kernel should be 1-bw: {} != {}",
        k,
        1.0 - 0.1
    );
}

#[test]
fn test_aitchison_aitken_different_category() {
    let dom = Cat::new(["a", "b", "c"], Uniform);

    let sctx = AitchisonAitkenKernel::get_scontext(0.1, &dom);

    let k = AitchisonAitkenKernel::compute(&"a".to_string(), &"b".to_string(), &(), &sctx, &dom);
    // Formula: bw / (|D| - 1) = 0.1 / 2 = 0.05
    let expected = 0.1 / 2.0;
    assert!(
        (k - expected).abs() < EPS,
        "Different-category kernel should be bw/(C-1): {} != {}",
        k,
        expected
    );
}

#[test]
fn test_aitchison_aitken_prior_uniform() {
    let dom = Cat::new(["a", "b", "c"], Uniform);
    let p = AitchisonAitkenKernel::prior(&"a".to_string(), &dom);
    // Formula: 1 / |D| = 1/3
    let expected = 1.0 / 3.0;
    assert!(
        (p - expected).abs() < EPS,
        "Prior should be uniform 1/C: {} != {}",
        p,
        expected
    );
}

#[test]
fn test_aitchison_aitken_sample_stays_in_domain() {
    let dom = Cat::new(["a", "b", "c"], Uniform);
    let mut rng = rand::rng();

    let sctx = AitchisonAitkenKernel::get_scontext(1.0, &dom);

    for _ in 0..30 {
        let sample = AitchisonAitkenKernel::sample(&mut rng, &"a".to_string(), &(), &sctx, &dom);
        assert!(
            dom.contains(&sample),
            "Sample must be a valid category: {}",
            sample
        );
    }
}

#[test]
fn test_aitchison_aitken_sums_to_one() {
    // Total probability across all categories should sum to 1 when categories are uniform
    let dom = Cat::new(["a", "b", "c"], Uniform);
    let x2 = "a".to_string();

    let sctx = AitchisonAitkenKernel::get_scontext(0.2, &dom);

    let total: f64 = dom
        .values
        .iter()
        .map(|x1| AitchisonAitkenKernel::compute(x1, &x2, &(), &sctx, &dom))
        .sum();
    assert!(
        (total - 1.0).abs() < EPS,
        "Aitchison-Aitken kernel must sum to 1 across categories: {}",
        total
    );
}

// ---- acquisition ----

#[test]
fn test_acquisition_ratio() {
    let acq = acquisition(2.0, 1.0);
    assert!(
        (acq - 2.0).abs() < EPS,
        "acquisition should be best/worse: {}",
        acq
    );
}

#[test]
fn test_acquisition_equal_densities() {
    let acq = acquisition(1.0, 1.0);
    assert!(
        (acq - 1.0).abs() < EPS,
        "Equal densities should give acquisition=1: {}",
        acq
    );
}

#[test]
fn test_acquisition_higher_good_pdf_is_better() {
    let acq_high = acquisition(3.0, 1.0);
    let acq_low = acquisition(0.5, 1.0);
    assert!(
        acq_high > acq_low,
        "Higher good_pdf should produce higher acquisition: {} <= {}",
        acq_high,
        acq_low
    );
}
