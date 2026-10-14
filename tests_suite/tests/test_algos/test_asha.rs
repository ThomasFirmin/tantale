use tantale::algos::bayesian::bandwidth::Optuna;
use tantale::algos::{
    Asha, LinearSplit, RandomSearch, Tpe, UniformWeighter, Univariate, asha, tpe,
};
use tantale::core::Experiment;
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack, Runable,
    stop::Evaluated,
};

use crate::cleaner::Cleaner;
use crate::init_func::sp_evaluator_sh;
use crate::run_checker::{run_reader, run_reader_eps};

#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_asha_run_seq");

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 1.61_f64.powi(i))
        .take_while(|&b| b <= 5.)
        .collect();
    if *budgets.last().unwrap() != 5. {
        let last = budgets.last_mut().unwrap();
        *last = 5.;
    }

    let pb = sp_evaluator_sh::get_problem();
    
    let sampler = RandomSearch::new();
    let opt =  Asha::new(sampler, 1., 5., 1.61).unwrap(); // log(max/min)

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_asha_run_seq");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    // 200 = 4 steps * 50 calls  + 6 evals for rungs filling
    run_reader("tmp_test_asha_run_seq", 200 + 6);

    let pb = sp_evaluator_sh::get_problem();

    let config = FolderConfig::new("tmp_test_asha_run_seq");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, asha!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);

    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budgets, budgets
    );
    assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
    exp.run();

    let pb = sp_evaluator_sh::get_problem();

    let config = FolderConfig::new("tmp_test_asha_run_seq");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, asha!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);
    // 400 = 4 steps * 100 calls  + 6 evals for rungs filling
    run_reader("tmp_test_asha_run_seq", 400 + 6);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budgets, budgets
    );
    assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
}

#[test]
fn test_fid_seq_run_tpe() {
    let _clean = Cleaner::new("tmp_test_asha_run_tpe");

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 1.61_f64.powi(i))
        .take_while(|&b| b <= 5.)
        .collect();
    if *budgets.last().unwrap() != 5. {
        let last = budgets.last_mut().unwrap();
        *last = 5.;
    }

    let pb = sp_evaluator_sh::get_problem();

    let sampler = Tpe::new(
        5,
        30,
        (Univariate, Optuna::new(false)),
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let opt = Asha::new(sampler, 1., 5., 1.61).unwrap(); // log(max/min)

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_asha_run_tpe");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    // 200 = 4 steps * 50 calls  + 6 evals for rungs filling
    run_reader("tmp_test_asha_run_tpe", 200 + 6);

    let pb = sp_evaluator_sh::get_problem();

    let config = FolderConfig::new("tmp_test_asha_run_tpe");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, asha!(tpe!(Univariate, Optuna, UniformWeighter, LinearSplit)), Evaluated, _, _>::load(pb, rec, check);

    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budgets, budgets
    );
    assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
    exp.run();

    let pb = sp_evaluator_sh::get_problem();

    let config = FolderConfig::new("tmp_test_asha_run_tpe");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, asha!(tpe!(Univariate, Optuna, UniformWeighter, LinearSplit)), Evaluated, _, _>::load(pb, rec, check);
    // 400 = 4 steps * 100 calls  + 6 evals for rungs filling
    run_reader("tmp_test_asha_run_tpe", 400 + 6);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budgets, budgets
    );
    assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
}

// #[test]
// fn test_fid_seq_parrun() {
//     let _clean = Cleaner::new("tmp_test_asha_parrun");

//     let mut budgets: Vec<f64> = (0..)
//         .map(|i| 1.61_f64.powi(i))
//         .take_while(|&b| b <= 5.)
//         .collect();
//     if *budgets.last().unwrap() != 5. {
//         let last = budgets.last_mut().unwrap();
//         *last = 5.;
//     }

//     let sp = sp_evaluator_sh::get_searchspace();
//     let obj = sp_evaluator_sh::get_function();
//     let sampler = RandomSearch::new();
//     let opt = Asha::new(sampler, 1., 5., 1.61).unwrap();

//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_asha_parrun").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     // 200 = 4 steps * 50 calls  + 6 evals for rungs filling, epsilon added to account for parallelism
//     run_reader_eps("tmp_test_asha_parrun", 200 + 6, num_cpus::get() * 3);

//     let sp = sp_evaluator_sh::get_searchspace();
//     let obj = sp_evaluator_sh::get_function();

//     let config = FolderConfig::new("tmp_test_asha_parrun").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         asha!(RandomSearch),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );

//     let expstop: &mut Evaluated = exp.mut_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 50 && expstop.calls() <= max_call,
//         "Number of calls is wrong, it should be between 50 and {}",
//         max_call
//     );
//     expstop.1 = 100;
//     let expoptimizer = exp.get_mut_optimizer();
//     assert_eq!(
//         expoptimizer.budgets, budgets,
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budgets, budgets
//     );
//     assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");

//     exp.run();

//     let sp = sp_evaluator_sh::get_searchspace();
//     let obj = sp_evaluator_sh::get_function();

//     let config = FolderConfig::new("tmp_test_asha_parrun").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(
//         threaded,
//         asha!(RandomSearch),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );
//     // 400 = 4 steps * 50 calls  + 6 evals for rungs filling, epsilon added to account for parallelism
//     run_reader_eps("tmp_test_asha_parrun", 400 + 6, num_cpus::get() * 3);
//     let expstop: &Evaluated = exp.get_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 100 && expstop.calls() <= max_call,
//         "Number of calls is wrong, it should be between 100 and {}",
//         max_call
//     );
//     let expoptimizer = exp.sampler();
//     assert_eq!(
//         expoptimizer.budgets, budgets,
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budgets, budgets
//     );
//     assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
// }

// #[test]
// fn test_fid_seq_parrun_tpe() {
//     let _clean = Cleaner::new("tmp_test_asha_parrun_tpe");

//     let mut budgets: Vec<f64> = (0..)
//         .map(|i| 1.61_f64.powi(i))
//         .take_while(|&b| b <= 5.)
//         .collect();
//     if *budgets.last().unwrap() != 5. {
//         let last = budgets.last_mut().unwrap();
//         *last = 5.;
//     }

//     let sp = sp_evaluator_sh::get_searchspace();
//     let obj = sp_evaluator_sh::get_function();
//     let sampler = Tpe::new(
//         5,
//         30,
//         (Univariate, Optuna::new(false)),
//         UniformWeighter::default(),
//         LinearSplit::new(0.25).unwrap(),
//     );
//     let opt = Asha::new(sampler, 1., 5., 1.61).unwrap();

//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_asha_parrun_tpe").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     // 200 = 4 steps * 50 calls  + 6 evals for rungs filling, epsilon added to account for parallelism
//     run_reader_eps("tmp_test_asha_parrun_tpe", 200 + 6, num_cpus::get() * 3);

//     let sp = sp_evaluator_sh::get_searchspace();
//     let obj = sp_evaluator_sh::get_function();

//     let config = FolderConfig::new("tmp_test_asha_parrun_tpe").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         asha!(tpe!(Univariate, Optuna, UniformWeighter, LinearSplit)),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );

//     let expstop: &mut Evaluated = exp.mut_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 50 && expstop.calls() <= max_call,
//         "Number of calls is wrong, it should be between 50 and {}",
//         max_call
//     );
//     expstop.1 = 100;
//     let expoptimizer = exp.get_mut_optimizer();
//     assert_eq!(
//         expoptimizer.budgets, budgets,
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budgets, budgets
//     );
//     assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");

//     exp.run();

//     let sp = sp_evaluator_sh::get_searchspace();
//     let obj = sp_evaluator_sh::get_function();

//     let config = FolderConfig::new("tmp_test_asha_parrun_tpe").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(
//         threaded,
//         asha!(tpe!(Univariate, Optuna, UniformWeighter, LinearSplit)),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );
//     // 400 = 4 steps * 50 calls  + 6 evals for rungs filling, epsilon added to account for parallelism
//     run_reader_eps("tmp_test_asha_parrun_tpe", 400 + 6, num_cpus::get() * 3);
//     let expstop: &Evaluated = exp.get_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 100 && expstop.calls() <= max_call,
//         "Number of calls is wrong, it should be between 100 and {}",
//         max_call
//     );
//     let expoptimizer = exp.sampler();
//     assert_eq!(
//         expoptimizer.budgets, budgets,
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budgets, budgets
//     );
//     assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
// }
