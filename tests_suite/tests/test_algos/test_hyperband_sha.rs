use tantale::{
    algos::{BatchRandomSearch, batchrandomsearch, hyperband}, core::{
        CSVRecorder, FolderConfig, MessagePack, experiment::{Experiment, Runable}, stop::Evaluated,
    },
};

use tantale::algos::{Hyperband, Sha, sha};

use crate::cleaner::Cleaner;
use crate::init_func::sp_evaluator_sh;
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

    let pb = sp_evaluator_sh::get_problem();

    let sampler = BatchRandomSearch::new(10);
    let opt = Sha::new(sampler, 1., 5., 1.61).unwrap().into_hyperband();

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_hyperband_sha_run");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    // 3550 = 4 brackets at
    // n = 1*1.61^3 = 5 |-> 4 |-> 4 |-> 4 (batch sizes of bracket 1) | s = [3,2,1,0]
    // bracket 1 : 10 |-> bracket 2 : 6 (10/1.61)
    // bracket 1 : 10 |-> bracket 2 : 6 (10/1.61)
    // bracket 1 : 10 |-> bracket 2 : 6 (10/1.61) |-> bracket 3 : 3 (6/1.61)
    // bracket 1 : 10 |-> bracket 2 : 6 (10/1.61) |-> bracket 3 : 3 (6/1.61) |-> bracket 4 : 1 (3/1.61) (fully evaluated)
    // 10+6+10+6+10+6+3+10+6+3+1 = 71 partial evaluations to get 1 full
    // So 71 * 50 = 3550 evaluations to get 50 full evaluations
    run_reader("tmp_test_hyperband_sha_run", 3550);

    let pb = sp_evaluator_sh::get_problem();


    let config = FolderConfig::new("tmp_test_hyperband_sha_run");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, hyperband!(sha!(batchrandomsearch!())), Evaluated, _, _>::load(pb, rec, check);

    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.budget_min,
        *budgets.first().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budget_min,
        budgets.first().unwrap()
    );
    assert_eq!(
        expoptimizer.budget_max,
        *budgets.last().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budget_max,
        budgets.last().unwrap()
    );
    assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
    assert_eq!(
        expoptimizer.scaling, expoptimizer.inner.scaling,
        "Scaling factor is not equal to inner scaling factor"
    );
    assert_eq!(
        expoptimizer.budget_min, expoptimizer.inner.budgets[0],
        "Budget min is not equal to inner budget min"
    );
    let current_max = 5. * 1.61_f64.powi(-(expoptimizer.current_s as i32));
    assert_eq!(
        current_max,
        *expoptimizer.inner.budgets.last().unwrap(),
        "Current max budget is not equal to inner budget max"
    );
    exp.run();

    let pb = sp_evaluator_sh::get_problem();


    let config = FolderConfig::new("tmp_test_hyperband_sha_run");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, hyperband!(sha!(batchrandomsearch!())), Evaluated, _, _>::load(pb, rec, check);

    run_reader("tmp_test_hyperband_sha_run", 7098);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.budget_min,
        *budgets.first().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budget_min,
        budgets.first().unwrap()
    );
    assert_eq!(
        expoptimizer.budget_max,
        *budgets.last().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budget_max,
        budgets.last().unwrap()
    );
    assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
    assert_eq!(
        expoptimizer.scaling, expoptimizer.inner.scaling,
        "Scaling factor is not equal to inner scaling factor"
    );
    assert_eq!(
        expoptimizer.budget_min, expoptimizer.inner.budgets[0],
        "Budget min is not equal to inner budget min"
    );
    let current_max = 5. * 1.61_f64.powi(-(expoptimizer.current_s as i32));
    assert_eq!(
        current_max,
        *expoptimizer.inner.budgets.last().unwrap(),
        "Current max budget is not equal to inner budget max"
    );
}

// #[test]
// fn test_fid_seq_parrun() {
//     let _clean = Cleaner::new("tmp_test_hyperband_sha_parrun");

//     let mut budgets: Vec<f64> = (0..)
//         .map(|i| 1.61_f64.powi(i))
//         .take_while(|&b| b <= 5.)
//         .collect();
//     //If final budget is not budget_max, modify final budget to be budget_max
//     if *budgets.last().unwrap() != 5. {
//         let last = budgets.last_mut().unwrap();
//         *last = 5.;
//     }

//     let pb = sp_evaluator_sh::get_problem();

//     let sampler = BatchRandomSearch::new(10);
//     let sha = Sha::new(sampler, 10, 1., 5., 1.61).unwrap();
//     let opt = Hyperband::new(sha);

//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_hyperband_sha_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     run_reader("tmp_test_hyperband_sha_parrun", 1450);

//     let pb = sp_evaluator_sh::get_problem();


//     let config = FolderConfig::new("tmp_test_hyperband_sha_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         hyperband!(sha!(BatchRandomSearch)),
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
//         expoptimizer.budget_min,
//         *budgets.first().unwrap(),
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budget_min,
//         budgets.first().unwrap()
//     );
//     assert_eq!(
//         expoptimizer.budget_max,
//         *budgets.last().unwrap(),
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budget_max,
//         budgets.last().unwrap()
//     );
//     assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
//     assert_eq!(
//         expoptimizer.scaling, expoptimizer.inner.scaling,
//         "Scaling factor is not equal to inner scaling factor"
//     );
//     assert_eq!(
//         expoptimizer.budget_min, expoptimizer.inner.budgets[0],
//         "Budget min is not equal to inner budget min"
//     );
//     let current_max = 5. * 1.61_f64.powi(-(expoptimizer.current_s as i32));
//     assert_eq!(
//         current_max,
//         *expoptimizer.inner.budgets.last().unwrap(),
//         "Current max budget is not equal to inner budget max"
//     );

//     exp.run();

//     let pb = sp_evaluator_sh::get_problem();


//     let config = FolderConfig::new("tmp_test_hyperband_sha_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(
//         threaded,
//         hyperband!(sha!(BatchRandomSearch)),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );
//     run_reader("tmp_test_hyperband_sha_parrun", 2900);
//     let expstop: &Evaluated = exp.get_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 100 && expstop.calls() <= max_call,
//         "Number of calls is wrong, it should be between 100 and {}",
//         max_call
//     );
//     let expoptimizer = exp.sampler();
//     assert_eq!(
//         expoptimizer.budget_min,
//         *budgets.first().unwrap(),
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budget_min,
//         budgets.first().unwrap()
//     );
//     assert_eq!(
//         expoptimizer.budget_max,
//         *budgets.last().unwrap(),
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budget_max,
//         budgets.last().unwrap()
//     );
//     assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
//     assert_eq!(
//         expoptimizer.scaling, expoptimizer.inner.scaling,
//         "Scaling factor is not equal to inner scaling factor"
//     );
//     assert_eq!(
//         expoptimizer.budget_min, expoptimizer.inner.budgets[0],
//         "Budget min is not equal to inner budget min"
//     );
//     let current_max = 5. * 1.61_f64.powi(-(expoptimizer.current_s as i32));
//     assert_eq!(
//         current_max,
//         *expoptimizer.inner.budgets.last().unwrap(),
//         "Current max budget is not equal to inner budget max"
//     );
// }
