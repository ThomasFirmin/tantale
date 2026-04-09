use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack, SaverConfig,
    experiment::{Runable, mono, threaded},
    load,
    stop::Evaluated,
};

use tantale::algos::{Hyperband, Sha, sha};

use crate::init_func::{sp_evaluator_sh, FidOutEvaluator};
use crate::cleaner::Cleaner;
use crate::run_checker::run_reader;

#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_hyperband_sha_run");

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
    let sha = Sha::new(10, 1., 5., 1.61); // log(max/min)
    let opt = Hyperband::new(sha);
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_hyperband_sha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    // 1450 = 4 brackets at
    // n = 1*1.61^3 = 5 |-> 4 |-> 4 |-> 4 (batch sizes of bracket 1) | s = [3,2,1,0]
    // bracket 1 : 5 |-> bracket 2 : 3 (5/1.61)
    // bracket 1 : 4 |-> bracket 2 : 2 (4/1.61)
    // bracket 1 : 4 |-> bracket 2 : 2 (4/1.61) |-> bracket 3 : 1 (2/1.61)
    // bracket 1 : 4 |-> bracket 2 : 2 (4/1.61) |-> bracket 3 : 1 (2/1.61) |-> bracket 4 : 1 (1/1.61) (fully evaluated)
    // 5+3+4+2+4+2+1+4+2+1+1 = 29 partial evaluations to get 1 full
    // So 29 * 50 = 1450 evaluations to get 50 full evaluations
    run_reader("tmp_test_hyperband_sha_run", 1450);

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_hyperband_sha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        mono,
        Hyperband<Sha, _, _, _>,
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
        expoptimizer.0.budget_min, expoptimizer.0.inner.0.budgets[0],
        "Budget min is not equal to inner budget min"
    );
    let current_max = 5. * 1.61_f64.powi(-(expoptimizer.0.current_s as i32));
    assert_eq!(
        current_max, *expoptimizer.0.inner.0.budgets.last().unwrap(),
        "Current max budget is not equal to inner budget max"
    );
    exp.run();

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_hyperband_sha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(
        mono,
        Hyperband<Sha, _, _, _>,
        Evaluated,
        (sp, cod),
        obj,
        (rec, check)
    );
    run_reader("tmp_test_hyperband_sha_run", 2900);
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
        expoptimizer.0.budget_min, expoptimizer.0.inner.0.budgets[0],
        "Budget min is not equal to inner budget min"
    );
    let current_max = 5. * 1.61_f64.powi(-(expoptimizer.0.current_s as i32));
    assert_eq!(
        current_max, *expoptimizer.0.inner.0.budgets.last().unwrap(),
        "Current max budget is not equal to inner budget max"
    );
}

#[test]
fn test_fid_seq_parrun() {
    let _clean = Cleaner::new("tmp_test_hyperband_sha_parrun");

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
    let sha = Sha::new(10, 1., 5., 1.61);
    let opt = Hyperband::new(sha);
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_hyperband_sha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_hyperband_sha_parrun", 1450);

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_hyperband_sha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        threaded,
        Hyperband<Sha, _, _, _>,
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
        expoptimizer.0.budget_min, expoptimizer.0.inner.0.budgets[0],
        "Budget min is not equal to inner budget min"
    );
    let current_max = 5. * 1.61_f64.powi(-(expoptimizer.0.current_s as i32));
    assert_eq!(
        current_max, *expoptimizer.0.inner.0.budgets.last().unwrap(),
        "Current max budget is not equal to inner budget max"
    );

    exp.run();

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_hyperband_sha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(
        threaded,
        Hyperband<Sha, _, _, _>,
        Evaluated,
        (sp, cod),
        obj,
        (rec, check)
    );
    run_reader("tmp_test_hyperband_sha_parrun", 2900);
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
        expoptimizer.0.budget_min, expoptimizer.0.inner.0.budgets[0],
        "Budget min is not equal to inner budget min"
    );
    let current_max = 5. * 1.61_f64.powi(-(expoptimizer.0.current_s as i32));
    assert_eq!(
        current_max, *expoptimizer.0.inner.0.budgets.last().unwrap(),
        "Current max budget is not equal to inner budget max"
    );
}
