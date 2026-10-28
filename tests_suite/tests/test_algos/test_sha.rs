use serial_test::serial;
use tantale::algos::{BatchRandomSearch, RandomSearch, Sha, batchrandomsearch, sha};
use tantale::core::{Experiment, HasId};
use tantale::core::optimizer::opt::Batched;
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack,
    experiment::Runable,
    stop::Evaluated,
};

use crate::cleaner::Cleaner;
use crate::init_func;
use crate::run_checker::run_reader;

pub mod searchspace {
    use super::init_func::{FidOutEvaluator, FnState};
    use tantale::core::Step;
    use tantale::core::{Real, distribution::Uniform};
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 5;

    objective!(
        pub fn example() -> (FidOutEvaluator, FnState) {
            let _a = [! a | Real(0.0, 10.0, Uniform) | !];
            let b = [! b | Real(0.0, 100.0, Uniform) | !];
            let c = [! c | Real(0.0, 100.0, Uniform) | !];
            let d = [! d | Real(0.0, 100.0, Uniform) | !];
            let e = [! e | Real(0.0, 100.0, Uniform) | !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 4 {Step::Evaluated} else{Step::Partially(state.state)};

            let obj = if state.state == 1 { b } else if state.state == 2 { c } else if state.state == 3 { d } else if state.state == 4 { e } else {panic!("Invalid state")};

            (
                FidOutEvaluator{
                    obj,
                    fid: evalstate,
                },
                state
            )
        }
    );
}

#[test]
fn test_fid_batch_run() {
    let _clean = Cleaner::new("tmp_test_sha_batchrun");

    let pb = searchspace::get_problem();
    let sampler = RandomSearch::new();
    let opt = Sha::new(sampler, 1., 8., 2., 10).unwrap(); // log(max/min)

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_sha_batchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    // To get 2 fully evaluated solutions we need:
    // 10 Evaluations at fidelity 1
    // 6 Evaluations at fidelity 1.61 (n*1.61^-1)
    // 3 Evaluations at fidelity 2.59 (n*1.61^-2)
    // 2 Evaluations at fidelity 4.17 (5.) (n*1.61^-3)
    // To get 50 fully evaluated solutions we need 25 times the above, so 525 evaluations in total
    run_reader("tmp_test_sha_batchrun", 900);

    let pb = searchspace::get_problem();

    let config = FolderConfig::new("tmp_test_sha_batchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, sha!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);

    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.sampler();
    assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");

    exp.run();

    let pb = searchspace::get_problem();

    let config = FolderConfig::new("tmp_test_sha_batchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, sha!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);

    run_reader("tmp_test_sha_batchrun", 1800);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.sampler();
    assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");
}

// #[test]
// fn test_fid_batch_parrun() {
//     let _clean = Cleaner::new("tmp_test_sh_parrun");

//     let pb = searchspace::get_problem();
//     let sampler = BatchRandomSearch::new(10);
//     let opt = Sha::new(sampler, 1., 5., 1.61).unwrap();

//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_sh_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     run_reader("tmp_test_sh_parrun", 1000);

//     let sp = searchspace::get_searchspace();
//     let obj = searchspace::get_function();

//     let config = FolderConfig::new("tmp_test_sh_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         sha!(BatchRandomSearch),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );

//     let expstop = exp.mut_stop();
//     assert_eq!(expstop.0, 50, "Number of calls is wrong");
//     expstop.1 = 100;
//     let expoptimizer = exp.get_mut_optimizer();
//     assert_eq!(
//         expoptimizer.calls, 200,
//         "Number of iteration is wrong"
//     );
//     assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");

//     exp.run();

//     let sp = searchspace::get_searchspace();
//     let obj = searchspace::get_function();

//     let config = FolderConfig::new("tmp_test_sh_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(
//         threaded,
//         sha!(BatchRandomSearch),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );
//     run_reader("tmp_test_sh_parrun", 2000);
//     let expstop = exp.get_stop();
//     assert_eq!(expstop.0, 100, "Number of calls is wrong");
//     let expoptimizer = exp.sampler();
//     assert_eq!(
//         expoptimizer.calls, 400,
//         "Number of iteration is wrong"
//     );
//     assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");
// }
