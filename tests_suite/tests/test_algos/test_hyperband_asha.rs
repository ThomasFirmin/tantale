use tantale::algos::{Asha, Hyperband, asha};
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack, SaverConfig,
    experiment::{Runable, mono, threaded},
    load,
    stop::Evaluated,
};

use crate::cleaner::Cleaner;
use crate::init_func::{FidOutEvaluator, sp_evaluator_sh};
use crate::run_checker::{run_reader, run_reader_eps};

#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_hyperband_asha_run");

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 1.61_f64.powi(i))
        .take_while(|&b| b <= 5.)
        .collect();
    //If final budget is not budget_max, modify final budget to be budget_max
    if *budgets.last().unwrap() != 5. {
        let last = budgets.last_mut().unwrap();
        *last = 5.;
    }

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let asha = Asha::new(1., 5., 1.61); // log(max/min)
    let opt = Hyperband::new(asha);
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_hyperband_asha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    // 1000 =
    // s = 3 : 2 for 1st r (b = 1) + 1 at bmax (1.19) |rung|//1.61 != 0 | [1, 1.19]
    // s = 2 : 2 for 1st r (b = 1) + 1 at bmax (1.92) |rung|//1.61 != 0 | [1, 1.92]
    // s = 1 : 3 for 1st r (b = 1) + 2 for 2nd r (1.61) + 1 at bmax (3.10) |rung|//1.61 != 0 | [1, 1.61, 3.10]
    // s = 0 : 4 for 1st r (b = 1) + 3 for 2nd r (1.61) + 2 for 3nd r (2.59) + 1 at bmax (5) |rung|//1.61 != 0 | [1, 1.61, 2.59, 5]
    // 2 + 2 + 3 + 4 + 1 + 1 + 2 + 3 + 1 + 2 + 1 = 22
    // Total of 22 partial eval to get 1 full eval at bmax = 5
    // So 22 * 50 = 1100 calls to get 50 full eval at bmax
    run_reader("tmp_test_hyperband_asha_run", 1100);

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_hyperband_asha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        mono,
        Hyperband<Asha<_>, _, _, _>,
        Evaluated,
        (sp, cod),
        obj,
        (rec, check)
    );

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.budget_min,
        *budgets.first().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budget_min,
        budgets.first().unwrap()
    );
    assert_eq!(
        expoptimizer.0.budget_max,
        *budgets.last().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budget_max,
        budgets.last().unwrap()
    );
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");
    assert_eq!(
        expoptimizer.0.scaling, expoptimizer.0.inner.0.scaling,
        "Scaling factor is not equal to inner scaling factor"
    );
    assert_eq!(
        expoptimizer.0.budget_min,
        *expoptimizer.0.inner.0.budgets.first().unwrap(),
        "Budget min is not equal to inner budget min"
    );
    let current_max = 5. * 1.61_f64.powi(-(expoptimizer.0.current_s as i32));
    assert_eq!(
        current_max,
        *expoptimizer.0.inner.0.budgets.last().unwrap(),
        "Current max budget is not equal to inner budget max"
    );
    exp.run();

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_hyperband_asha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(
        mono,
        Hyperband<Asha<_>, _, _, _>,
        Evaluated,
        (sp, cod),
        obj,
        (rec, check)
    );

    run_reader("tmp_test_hyperband_asha_run", 2200);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.budget_min,
        *budgets.first().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budget_min,
        budgets.first().unwrap()
    );
    assert_eq!(
        expoptimizer.0.budget_max,
        *budgets.last().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budget_max,
        budgets.last().unwrap()
    );
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");
    assert_eq!(
        expoptimizer.0.scaling, expoptimizer.0.inner.0.scaling,
        "Scaling factor is not equal to inner scaling factor"
    );
    assert_eq!(
        expoptimizer.0.budget_min,
        *expoptimizer.0.inner.0.budgets.first().unwrap(),
        "Budget min is not equal to inner budget min"
    );
    let current_max = 5. * 1.61_f64.powi(-(expoptimizer.0.current_s as i32));
    assert_eq!(
        current_max,
        *expoptimizer.0.inner.0.budgets.last().unwrap(),
        "Current max budget is not equal to inner budget max"
    );
}

#[test]
fn test_fid_seq_parrun() {
    let _clean = Cleaner::new("tmp_test_hyperband_asha_parrun");

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 1.61_f64.powi(i))
        .take_while(|&b| b <= 5.)
        .collect();
    //If final budget is not budget_max, modify final budget to be budget_max
    if *budgets.last().unwrap() != 5. {
        let last = budgets.last_mut().unwrap();
        *last = 5.;
    }

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let asha = Asha::new(1., 5., 1.61);
    let opt = Hyperband::new(asha);
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_hyperband_asha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader_eps(
        "tmp_test_hyperband_asha_parrun",
        1100,
        num_cpus::get() * 8 * 50,
    );

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_hyperband_asha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        threaded,
        Hyperband<Asha<_>, _, _, _>,
        Evaluated,
        (sp, cod),
        obj,
        (rec, check)
    );

    let expstop: &mut Evaluated = exp.get_mut_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(
        expstop.calls() >= 50 && expstop.calls() <= max_call,
        "Number of calls is wrong, it should be between 50 and {}",
        max_call
    );
    expstop.1 = 100;
    let expoptimizer = exp.get_mut_optimizer();
    assert_eq!(
        expoptimizer.0.budget_min,
        *budgets.first().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budget_min,
        budgets.first().unwrap()
    );
    assert_eq!(
        expoptimizer.0.budget_max,
        *budgets.last().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budget_max,
        budgets.last().unwrap()
    );
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");
    assert_eq!(
        expoptimizer.0.scaling, expoptimizer.0.inner.0.scaling,
        "Scaling factor is not equal to inner scaling factor"
    );
    assert_eq!(
        expoptimizer.0.budget_min,
        *expoptimizer.0.inner.0.budgets.first().unwrap(),
        "Budget min is not equal to inner budget min"
    );
    let current_max = 5. * 1.61_f64.powi(-(expoptimizer.0.current_s as i32));
    assert_eq!(
        current_max,
        *expoptimizer.0.inner.0.budgets.last().unwrap(),
        "Current max budget is not equal to inner budget max"
    );

    exp.run();

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_hyperband_asha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(
        threaded,
        Hyperband<Asha<_>, _, _, _>,
        Evaluated,
        (sp, cod),
        obj,
        (rec, check)
    );
    run_reader_eps(
        "tmp_test_hyperband_asha_parrun",
        2200,
        num_cpus::get() * 8 * 50,
    );
    let expstop: &Evaluated = exp.get_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(
        expstop.calls() >= 100 && expstop.calls() <= max_call,
        "Number of calls is wrong, it should be between 100 and {}",
        max_call
    );
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.budget_min,
        *budgets.first().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budget_min,
        budgets.first().unwrap()
    );
    assert_eq!(
        expoptimizer.0.budget_max,
        *budgets.last().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budget_max,
        budgets.last().unwrap()
    );
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");
    assert_eq!(
        expoptimizer.0.scaling, expoptimizer.0.inner.0.scaling,
        "Scaling factor is not equal to inner scaling factor"
    );
    assert_eq!(
        expoptimizer.0.budget_min,
        *expoptimizer.0.inner.0.budgets.first().unwrap(),
        "Budget min is not equal to inner budget min"
    );
    let current_max = 5. * 1.61_f64.powi(-(expoptimizer.0.current_s as i32));
    assert_eq!(
        current_max,
        *expoptimizer.0.inner.0.budgets.last().unwrap(),
        "Current max budget is not equal to inner budget max"
    );
}
