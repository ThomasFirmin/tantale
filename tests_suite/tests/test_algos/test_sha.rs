use tantale::algos::{BatchRandomSearch, Sha, batchrandomsearch, sha};
use tantale::core::Experiment;
use tantale::core::optimizer::opt::Batched;
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack,
    experiment::Runable,
    stop::Evaluated,
};

use crate::cleaner::Cleaner;
use crate::init_func::sp_evaluator_sh;
use crate::run_checker::run_reader;

#[test]
fn test_fid_batch_run() {
    let _clean = Cleaner::new("tmp_test_sha_batchrun");

    let pb = sp_evaluator_sh::get_problem();
    let sampler = BatchRandomSearch::new(10);
    let opt = Sha::new(sampler, 1., 5., 1.61).unwrap(); // log(max/min)

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_sha_batchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader("tmp_test_sha_batchrun", 1000);

    let pb = sp_evaluator_sh::get_problem();

    let config = FolderConfig::new("tmp_test_sha_batchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, sha!(batchrandomsearch!()), Evaluated, _, _>::load(pb, rec, check);

    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.sampler.calls, 51,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");

    exp.run();

    let pb = sp_evaluator_sh::get_problem();

    let config = FolderConfig::new("tmp_test_sha_batchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, sha!(batchrandomsearch!()), Evaluated, _, _>::load(pb, rec, check);

    run_reader("tmp_test_sha_batchrun", 1998);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.sampler.calls, 101,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");
}

// #[test]
// fn test_fid_batch_parrun() {
//     let _clean = Cleaner::new("tmp_test_sh_parrun");

//     let pb = sp_evaluator_sh::get_problem();
//     let sampler = BatchRandomSearch::new(10);
//     let opt = Sha::new(sampler, 1., 5., 1.61).unwrap();

//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_sh_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     run_reader("tmp_test_sh_parrun", 1000);

//     let sp = sp_evaluator_sh::get_searchspace();
//     let obj = sp_evaluator_sh::get_function();

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

//     let sp = sp_evaluator_sh::get_searchspace();
//     let obj = sp_evaluator_sh::get_function();

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
