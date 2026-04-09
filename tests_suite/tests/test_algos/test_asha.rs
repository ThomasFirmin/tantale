use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack, SaverConfig,
    experiment::{Runable, mono, threaded},
    load,
    stop::Evaluated,
};
use tantale::algos::{Asha, asha};

use crate::init_func::{sp_evaluator_sh, FidOutEvaluator};
use crate::cleaner::Cleaner;
use crate::run_checker::{run_reader, run_reader_eps};

#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_asha_run");

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 1.61_f64.powi(i))
        .take_while(|&b| b <= 5.)
        .collect();
    if *budgets.last().unwrap() != 5. {
        let last = budgets.last_mut().unwrap();
        *last = 5.;
    }

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let opt = Asha::new(1., 5., 1.61); // log(max/min)
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_asha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    // 200 = 4 steps * 50 calls  + 6 evals for rungs filling
    run_reader("tmp_test_asha_run", 200 + 6);

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_asha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, Asha<_>, Evaluated, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budgets, budgets
    );
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");
    exp.run();

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_asha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, Asha<_>, Evaluated, (sp, cod), obj, (rec, check));
    // 400 = 4 steps * 100 calls  + 6 evals for rungs filling
    run_reader("tmp_test_asha_run", 400 + 6);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budgets, budgets
    );
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");
}

#[test]
fn test_fid_seq_parrun() {
    let _clean = Cleaner::new("tmp_test_asha_parrun");

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 1.61_f64.powi(i))
        .take_while(|&b| b <= 5.)
        .collect();
    if *budgets.last().unwrap() != 5. {
        let last = budgets.last_mut().unwrap();
        *last = 5.;
    }

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let opt = Asha::new(1., 5., 1.61);
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_asha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    // 200 = 4 steps * 50 calls  + 6 evals for rungs filling, epsilon added to account for parallelism
    run_reader_eps("tmp_test_asha_parrun", 200 + 6, num_cpus::get() * 3);

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_asha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(threaded, Asha<_>, Evaluated, (sp, cod), obj, (rec, check));

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
        expoptimizer.0.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budgets, budgets
    );
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");

    exp.run();

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_asha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(threaded, Asha<_>, Evaluated, (sp, cod), obj, (rec, check));
    // 400 = 4 steps * 50 calls  + 6 evals for rungs filling, epsilon added to account for parallelism
    run_reader_eps("tmp_test_asha_parrun", 400 + 6, num_cpus::get() * 3);
    let expstop: &Evaluated = exp.get_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(
        expstop.calls() >= 100 && expstop.calls() <= max_call,
        "Number of calls is wrong, it should be between 100 and {}",
        max_call
    );
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budgets, budgets
    );
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");
}
