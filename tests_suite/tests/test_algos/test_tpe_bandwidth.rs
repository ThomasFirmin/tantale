use ndarray::Array2;
use num::Float;
use serde::{Deserialize, Serialize};
use tantale::Outcome;
use std::mem::MaybeUninit;
use std::sync::Arc;
use tantale::algos::bayesian::bandwidth::{
    Bandwidth, Hyperopt, Optuna, Scott, cat_bw, hyperopt_assign_col, magic_clip, optuna_bw, scott_col,
};
use tantale::core::{BaseSol, Bernoulli, Bool, Cat, ElemSingleCodomain, EmptyInfo, HasX, Id, SId, Sp, Uncomputed, Unit, Var, Xy};
use tantale::core::{Int, Nat, Real, sampler::Uniform};

const EPS: f64 = 1e-10;

// ---- optuna_bw ----

#[test]
fn test_optuna_bw_real() {
    let dom = Real::new(-5.0, 5.0, Uniform);
    let neg_div_dim_p_four = -1.0/(3 + 4) as f64;
    let bw = optuna_bw(3.0, neg_div_dim_p_four, &dom, false);
    // formula: (up - low)/5 * size^(-1/(dim + 4)) = 1.0/5 * 10^(-1/7)
    let expected = 10.0 / 5.0 * 3.0.powf(-1.0 / 7.0);
    assert!(
        (bw - expected).abs() < EPS,
        "optuna_bw Real failed: {} != {}",
        bw,
        expected
    );
}

#[derive(Outcome, Clone, Debug, Serialize, Deserialize)]
struct FakeOutcome {
    #[maximize]
    value: f64,
}

#[test]
fn test_optuna_object() {
    let mut bw = Optuna::new(false);

    let x1: Arc<[f64]> = vec![1.0, 1.0, 1.0].into_boxed_slice().into();
    let sol1: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::generate(), x1, EmptyInfo.into());
    let x2: Arc<[f64]> = vec![2.0, 2.0, 2.0].into_boxed_slice().into();
    let sol2: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::generate(), x2, EmptyInfo.into());
    let x3: Arc<[f64]> = vec![3.0, 3.0, 3.0].into_boxed_slice().into();
    let sol3: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::generate(), x3, EmptyInfo.into());

    let xy1 = Xy::new(sol1.clone_x(), ElemSingleCodomain::new(1.0).into());
    let xy2 = Xy::new(sol2.clone_x(), ElemSingleCodomain::new(2.0).into());
    let xy3 = Xy::new(sol3.clone_x(), ElemSingleCodomain::new(3.0).into());
    let xy = vec![&xy1, &xy2, &xy3];

    let scp = Sp::new(vec![
            Var::<Unit, Real>::new("a", Unit::new(Uniform), Real::new(-5.0, 5.0, Uniform)),
            Var::<Unit, Real>::new("a", Unit::new(Uniform), Real::new(-10.0, 10.0, Uniform)),
            Var::<Unit, Real>::new("a", Unit::new(Uniform), Real::new(-20.0, 20.0, Uniform)),
        ].into_boxed_slice()
    );

    let bw_vec = <Optuna as Bandwidth<Real, Sp<Unit, Real>, BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo, FakeOutcome>>::compute(&mut bw, xy.as_slice(), &scp);
    let bw_expected = vec![
        10./5. * 3.0.powf(-1./7.),
        20./5. * 3.0.powf(-1./7.),
        40./5. * 3.0.powf(-1./7.),
    ];

    assert_eq!(bw_vec, bw_expected, "optuna_bw vector computation failed: {:?} != {:?}", bw_vec, bw_expected);
}

#[test]
fn test_optuna_bw_int() {
    let dom = Int::new(-5, 5, Uniform);
    let neg_div_dim_p_four = -1.0/(3 + 4) as f64;
    let bw = optuna_bw(3.0, neg_div_dim_p_four, &dom, false);
    // formula: (up - low)/5 * size^(-1/(dim + 4)) = 1.0/5 * 10^(-1/7)
    let expected = 10.0 / 5.0 * 3.0.powf(-1.0 / 7.0);
    assert!(
        (bw - expected).abs() < EPS,
        "optuna_bw Int failed: {} != {}",
        bw,
        expected
    );
}

#[test]
fn test_optuna_bw_nat() {
    let dom = Nat::new(0, 10, Uniform);
    let neg_div_dim_p_four = -1.0/(3 + 4) as f64;
    let bw = optuna_bw(3.0, neg_div_dim_p_four, &dom, false);
    // formula: (up - low)/5 * size^(-1/(dim + 4)) = 1.0/5 * 10^(-1/7)
    let expected = 10.0 / 5.0 * 3.0.powf(-1.0 / 7.0);
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
    let neg_div_dim_p_four = -1.0/(3 + 4) as f64;
    let bw = optuna_bw(3.0, neg_div_dim_p_four, &dom, false);
    // formula: (up - low)/5 * size^(-1/(dim + 4)) = 1.0/5 * 10^(-1/7)
    let expected = 1.0 / 5.0 * 3.0.powf(-1.0 / 7.0);
    assert!(
        (bw - expected).abs() < EPS,
        "optuna_bw Unit failed: {} != {}",
        bw,
        expected
    );
}

#[test]
fn test_optuna_bw_with_clip_applies() {
    let dom = Real::new(0.0, 10.0, Uniform);
    // uml = (10-0)/5 = 2.0, unclipped bw = 2.0 * 200^(-1/7) is below the magic clip floor.
    let uml = 2.0;
    let unclipped = uml * 200.0_f64.powf(-1.0 / 7.0);
    let expected = magic_clip(unclipped, 200.0, uml);

    let bw = optuna_bw(200.0, -1.0 / 7.0, &dom, true);
    assert!(
        (bw - expected).abs() < EPS,
        "optuna_bw with clip failed: {} != {}",
        bw,
        expected
    );
}

// ---- cat_bw ----

#[test]
fn test_cat_bw_three_categories() {
    let dom = Cat::new(["a", "b", "c"], Uniform);
    let bw = cat_bw(10.0, &dom);
    // formula: (N + 1)/(N + C) = (10 + 1)/(10 + 3) = 11/13
    let expected = 11.0 / 13.0;
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
    let bw = cat_bw(8.0, &dom);
    // formula: (N + 1)/(N + C) = (8 + 1)/(8 + 2) = 9/10 = 0.9
    let expected = 9.0 / 10.0;
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
    let bw = cat_bw(8.0, &dom);
    // formula: (N + 1)/(N + C) = (8 + 1)/(8 + 2) = 9/10 = 0.9
    let expected = 9.0 / 10.0;
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
        [
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ],
        Uniform,
    );
    for n in [1.0, 5.0, 10.0, 100.0] {
        let bw = cat_bw(n, &dom);
        assert!(
            bw > 0.0 && bw < 1.0,
            "cat_bw out of (0,1): {} for n={}",
            bw,
            n
        );
    }
}

// ---- magic_clip ----

#[test]
fn test_magic_clip_clips_small_bandwidth() {
    // range=10, size=10, min(100, 10)=10, floor = range/10 = 1.0
    let bw_clipped = magic_clip(0.001, 10.0, 10.0);
    assert!(
        (bw_clipped - 1.0).abs() < EPS,
        "magic_clip should clip to 1.0, got: {}",
        bw_clipped
    );
}

#[test]
fn test_magic_clip_no_clip_for_large_bandwidth() {
    let bw_no_clip = magic_clip(5.0, 10.0, 10.0);
    assert!(
        (bw_no_clip - 5.0).abs() < EPS,
        "magic_clip should not clip 5.0, got: {}",
        bw_no_clip
    );
}

#[test]
fn test_magic_clip_large_size_uses_100_cap() {
    // range=100, size=200, min(100,200)=100, floor = 100.0/100 = 1.0
    let bw = magic_clip(0.5, 200.0, 100.0);
    assert!(
        (bw - 1.0).abs() < EPS,
        "magic_clip with large size should clip to 1.0, got: {}",
        bw
    );
}

// ---- hyperopt_assign_col (Hyperopt bandwidth) ----

#[test]
fn test_hyperopt_object() {
    let mut bw = Hyperopt::new(true, false);

    let x1: Arc<[f64]> = vec![1.0, 1.0, 1.0].into_boxed_slice().into();
    let sol1: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::generate(), x1, EmptyInfo.into());
    let x2: Arc<[f64]> = vec![3.0, 3.0, 3.0].into_boxed_slice().into();
    let sol2: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::generate(), x2, EmptyInfo.into());
    let x3: Arc<[f64]> = vec![6.0, 6.0, 6.0].into_boxed_slice().into();
    let sol3: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::generate(), x3, EmptyInfo.into());

    let xy1 = Xy::new(sol1.clone_x(), ElemSingleCodomain::new(1.0).into());
    let xy2 = Xy::new(sol2.clone_x(), ElemSingleCodomain::new(2.0).into());
    let xy3 = Xy::new(sol3.clone_x(), ElemSingleCodomain::new(3.0).into());
    let xy = vec![&xy1, &xy2, &xy3];

    let scp = Sp::new(vec![
            Var::<Unit, Real>::new("a", Unit::new(Uniform), Real::new(-10.0, 10.0, Uniform)),
            Var::<Unit, Real>::new("a", Unit::new(Uniform), Real::new(-10.0, 10.0, Uniform)),
            Var::<Unit, Real>::new("a", Unit::new(Uniform), Real::new(-10.0, 10.0, Uniform)),
        ].into_boxed_slice()
    );

    let bw_array = <Hyperopt as Bandwidth<Real, Sp<Unit, Real>, BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo, FakeOutcome>>::compute(&mut bw, xy.as_slice(), &scp);
    let bw_expected = vec![
        11.,
        3.,
        4.,
    ];
    for col in bw_array.columns(){
        let vec = col.to_vec();
        assert_eq!(vec, bw_expected, "Hyperopt vector computation failed: {:?} != {:?}", vec, bw_expected);

    }
}

#[test]
fn test_hyperopt_assign_col_no_endpoint() {
    let dom = Real::new(-10.0, 10.0, Uniform);
    let col = vec![1.0_f64, 3.0, 6.0];
    let mut res: Array2<MaybeUninit<f64>> = Array2::uninit((col.len(), 1));

    hyperopt_assign_col(col, 0, &dom, &mut res, false, false);
    let res = unsafe { res.assume_init() };

    // point 1: max(3-1, 1-1) = 2 ; point 3: max(6-3, 3-1) = 3 ; point 6: max(6-6, 6-3) = 3
    assert!((res[[0, 0]] - 2.0).abs() < EPS, "got {}", res[[0, 0]]);
    assert!((res[[1, 0]] - 3.0).abs() < EPS, "got {}", res[[1, 0]]);
    assert!((res[[2, 0]] - 3.0).abs() < EPS, "got {}", res[[2, 0]]);
}

#[test]
fn test_hyperopt_assign_col_with_endpoint() {
    let dom = Real::new(-10.0, 10.0, Uniform);
    let col = vec![1.0_f64, 3.0, 6.0];
    let mut res: Array2<MaybeUninit<f64>> = Array2::uninit((col.len(), 1));

    hyperopt_assign_col(col, 0, &dom, &mut res, true, false);
    let res = unsafe { res.assume_init() };

    // point 1: max(3-1, 1-0) = 2 ; point 3: max(6-3, 3-1) = 3 ; point 6: max(10-6, 6-3) = 4
    assert!((res[[0, 0]] - 11.0).abs() < EPS, "got {}", res[[0, 0]]);
    assert!((res[[1, 0]] - 3.0).abs() < EPS, "got {}", res[[1, 0]]);
    assert!((res[[2, 0]] - 4.0).abs() < EPS, "got {}", res[[2, 0]]);
}

#[test]
fn test_hyperopt_assign_col_clip_applies() {
    let dom = Real::new(0.0, 10.0, Uniform);
    let col = vec![1.0_f64, 3.0, 6.0];
    let mut res: Array2<MaybeUninit<f64>> = Array2::uninit((col.len(), 1));

    hyperopt_assign_col(col, 0, &dom, &mut res, false, true);
    let res = unsafe { res.assume_init() };

    // range=10, size=3 => floor = 10/3, which is above all unclipped bandwidths (2, 3, 3)
    let expected = 10.0 / 3.0;
    assert!((res[[0, 0]] - expected).abs() < EPS, "got {}", res[[0, 0]]);
    assert!((res[[1, 0]] - expected).abs() < EPS, "got {}", res[[1, 0]]);
    assert!((res[[2, 0]] - expected).abs() < EPS, "got {}", res[[2, 0]]);
}

// ---- scott_assign_col (Scott bandwidth) ----

#[test]
fn test_scott_object() {
    let mut bw = Scott::new(false);

    let x1: Arc<[f64]> = vec![1.0, 1.0, 1.0].into_boxed_slice().into();
    let sol1: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::generate(), x1, EmptyInfo.into());
    let x2: Arc<[f64]> = vec![2.0, 2.0, 2.0].into_boxed_slice().into();
    let sol2: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::generate(), x2, EmptyInfo.into());
    let x3: Arc<[f64]> = vec![3.0, 3.0, 3.0].into_boxed_slice().into();
    let sol3: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::generate(), x3, EmptyInfo.into());
    let x4: Arc<[f64]> = vec![4.0, 4.0, 4.0].into_boxed_slice().into();
    let sol4: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::generate(), x4, EmptyInfo.into());
    let x5: Arc<[f64]> = vec![5.0, 5.0, 5.0].into_boxed_slice().into();
    let sol5: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::generate(), x5, EmptyInfo.into());

    let xy1 = Xy::new(sol1.clone_x(), ElemSingleCodomain::new(1.0).into());
    let xy2 = Xy::new(sol2.clone_x(), ElemSingleCodomain::new(2.0).into());
    let xy3 = Xy::new(sol3.clone_x(), ElemSingleCodomain::new(3.0).into());
    let xy4 = Xy::new(sol4.clone_x(), ElemSingleCodomain::new(4.0).into());
    let xy5 = Xy::new(sol5.clone_x(), ElemSingleCodomain::new(5.0).into());
    let xy = vec![&xy1, &xy2, &xy3, &xy4, &xy5];

    let scp = Sp::new(vec![
            Var::<Unit, Real>::new("a", Unit::new(Uniform), Real::new(-100.0, 100.0, Uniform)),
            Var::<Unit, Real>::new("a", Unit::new(Uniform), Real::new(-100.0, 100.0, Uniform)),
            Var::<Unit, Real>::new("a", Unit::new(Uniform), Real::new(-100.0, 100.0, Uniform)),
        ].into_boxed_slice()
    );

    
    let bw_vec = <Scott as Bandwidth<Real, Sp<Unit, Real>, BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo, FakeOutcome>>::compute(&mut bw, xy.as_slice(), &scp);
    let bw_expected = vec![
        1.0854678306814147,
        1.0854678306814147,
        1.0854678306814147,
    ];

    assert_eq!(bw_vec, bw_expected, "Scott object vector computation failed: {:?} != {:?}", bw_vec, bw_expected);
}

#[test]
fn test_scott_assign_col_basic() {
    let dom = Real::new(0.0, 100.0, Uniform);
    let col = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let bw = scott_col(&col, &dom, false);
    
    // let arr = Array1::from(col.to_vec());
    // let std = arr.std(0.0);
    // let iqr = 2.0_f64;
    // let expected = 1.059 * 5.0_f64.powf(-0.2) * std.min(iqr / 1.34);
    let expected = 1.0854678306814147;

    assert!(
        (bw - expected).abs() < EPS,
        "scott_assign_col failed: {} != {}",
        bw,
        expected
    );
}

#[test]
fn test_scott_assign_col_clip_applies() {
    let dom = Real::new(0.0, 10.0, Uniform);
    let col = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let bw = scott_col(&col, &dom, true);

    // unclipped bandwidth is ~1.085, but range=10, size=5 => floor = 10/5 = 2.0
    assert!(
        (bw - 2.0).abs() < EPS,
        "scott_assign_col should clip to 2.0, got: {}",
        bw
    );
}
