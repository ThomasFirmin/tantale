use tantale::algos::batchrandomsearch;
use tantale::algos::{BatchRandomSearch, random_search::RandomSearch};
use tantale::core::Experiment;
use tantale::core::problem::Pb;
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack,
    experiment::{PoolMode, Runable},
    stop::Calls,
};

use crate::cleaner::Cleaner;
use crate::init_func::sp_evaluator_fid;
use crate::run_checker::{run_reader, run_reader_eps};

#[test]
fn test_fid_batch_run() {
    let _clean = Cleaner::new("tmp_test_fidbatchrun");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let opt: BatchRandomSearch<_> = BatchRandomSearch::new(7);

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_fidbatchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader_eps("tmp_test_fidbatchrun", 264, 20);

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);
    
    let config = FolderConfig::new("tmp_test_fidbatchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, batchrandomsearch!(),Calls,_,_>::load(pb, rec, check);

    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.mut_sampler();
    assert!(
        (expoptimizer.calls >= 39) && (expoptimizer.calls <= 41),
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_fidbatchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, batchrandomsearch!(),Calls,_,_>::load(pb, rec, check);
    run_reader_eps("tmp_test_fidbatchrun", 524, 50);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.sampler();
    assert_eq!(expoptimizer.calls, 81, "Number of iteration is wrong");
    assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");
}

#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_fidseqrun");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let opt = RandomSearch::new();

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_fidseqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();
    run_reader("tmp_test_fidseqrun", 250);

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_fidseqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, RandomSearch, Calls,_,_>::load(pb, rec, check);

    let expstop: &mut Calls = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_fidseqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, RandomSearch,Calls,_,_>::load(pb, rec, check);
    run_reader("tmp_test_fidseqrun", 500);
    let expstop: &Calls = exp.get_stop();
    assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
}


#[test]
fn test_fid_seq_run_loadpool() {
    let _clean = Cleaner::new("tmp_test_fidseqrun_loadpool");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let opt = RandomSearch::new();

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_fidseqrun_loadpool");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check).with_pool(PoolMode::Persistent);
    exp.run();
    run_reader("tmp_test_fidseqrun_loadpool", 250);

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_fidseqrun_loadpool");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, RandomSearch, Calls,_,_>::load(pb, rec, check).with_pool(PoolMode::Persistent);

    let expstop: &mut Calls = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_fidseqrun_loadpool");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, RandomSearch,Calls,_,_>::load(pb, rec, check).with_pool(PoolMode::Persistent);
    run_reader("tmp_test_fidseqrun_loadpool", 500);
    let expstop: &Calls = exp.get_stop();
    assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
}

#[test]
fn test_fid_batch_run_loadpool() {
    let _clean = Cleaner::new("tmp_test_fidbatchrun_loadpool");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let opt: BatchRandomSearch<_> = BatchRandomSearch::new(7);

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_fidbatchrun_loadpool");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check).with_pool(PoolMode::Persistent);
    exp.run();

    run_reader_eps("tmp_test_fidbatchrun_loadpool", 264, 20);

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);
    
    let config = FolderConfig::new("tmp_test_fidbatchrun_loadpool");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, batchrandomsearch!(),Calls,_,_>::load(pb, rec, check).with_pool(PoolMode::Persistent);

    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.mut_sampler();
    assert!(
        (expoptimizer.calls >= 39) && (expoptimizer.calls <= 41),
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_fidbatchrun_loadpool");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, batchrandomsearch!(),Calls,_,_>::load(pb, rec, check);
    run_reader_eps("tmp_test_fidbatchrun_loadpool", 524, 50);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.sampler();
    assert_eq!(expoptimizer.calls, 81, "Number of iteration is wrong");
    assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");
}

// #[test]
// fn test_fid_batch_parrun() {
//     let _clean = Cleaner::new("tmp_test_fidbatchparrun");

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();
//     let opt = BatchRandomSearch::new(7);

//     let stop = Calls::new(50);
//     let config = FolderConfig::new("tmp_test_fidbatchparrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     run_reader_eps("tmp_test_fidbatchparrun", 250, 35);

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();

//     let config = FolderConfig::new("tmp_test_fidbatchparrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(threaded, BatchRandomSearch, Calls, sp, obj, (rec, check));

//     let expstop = exp.get_mut_stop();
//     assert!(
//         expstop >= 50 && expstop <= 57,
//         "Number of calls is wrong"
//     );
//     expstop.1 = 100;
//     let expoptimizer = exp.get_mut_optimizer();
//     assert_eq!(expoptimizer.calls, 41, "Number of iteration is wrong");
//     assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");

//     exp.run();

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();

//     let config = FolderConfig::new("tmp_test_fidbatchparrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(threaded, BatchRandomSearch, Calls, sp, obj, (rec, check));
//     run_reader_eps("tmp_test_fidbatchparrun", 500, 35);
//     let expstop = exp.get_stop();
//     assert!(
//         expstop >= 100 && expstop <= 107,
//         "Number of calls is wrong"
//     );
//     let expoptimizer = exp.get_optimizer();
//     assert_eq!(expoptimizer.calls, 76, "Number of iteration is wrong");
//     assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");
// }

// #[test]
// fn test_fid_thr_seq_run() {
//     let _clean = Cleaner::new("tmp_test_fidthrseqrun");

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();
//     let opt = RandomSearch::new();
//     let stop = Calls::new(50);
//     let config = FolderConfig::new("tmp_test_fidthrseqrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();
//     run_reader_eps("tmp_test_fidthrseqrun", 250, 249);

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();

//     let config = FolderConfig::new("tmp_test_fidthrseqrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(threaded, RandomSearch, Calls, sp, obj, (rec, check));

//     let expstop: &mut Calls = exp.get_mut_stop();
//     let max_calls = 50 + num_cpus::get();
//     assert!(
//         expstop.calls() >= 50 && expstop.calls() <= max_calls,
//         "Number of calls is wrong, should be between 50 and {}",
//         max_calls
//     );
//     expstop.add(50);

//     exp.run();

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();

//     let config = FolderConfig::new("tmp_test_fidthrseqrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(threaded, RandomSearch, Calls, sp, obj, (rec, check));
//     run_reader_eps("tmp_test_fidthrseqrun", 500, 249);
//     let expstop: &Calls = exp.get_stop();
//     let max_calls = 100 + 2 * num_cpus::get();
//     assert!(
//         expstop.calls() >= 100 && expstop.calls() <= max_calls,
//         "Number of calls is wrong, should be between 50 and {}",
//         max_calls
//     );
// }

// #[test]
// fn test_fid_batch_parrun_loadpool() {
//     let _clean = Cleaner::new("tmp_test_fidbatchparrun_loadpool");

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();
//     let opt = BatchRandomSearch::new(7);

//     let stop = Calls::new(50);
//     let config = FolderConfig::new("tmp_test_fidbatchparrun_loadpool");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded_with_pool(sp, obj, opt, stop, (rec, check), PoolMode::Persistent);
//     exp.run();

//     run_reader_eps("tmp_test_fidbatchparrun_loadpool", 250, 35);

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();

//     let config = FolderConfig::new("tmp_test_fidbatchparrun_loadpool");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         PoolMode::Persistent,
//         BatchRandomSearch,
//         Calls,
//         sp,
//         obj,
//         (rec, check)
//     );

//     let expstop = exp.get_mut_stop();
//     assert!(
//         expstop >= 50 && expstop <= 57,
//         "Number of calls is wrong"
//     );
//     expstop.1 = 100;
//     let expoptimizer = exp.get_mut_optimizer();
//     assert_eq!(expoptimizer.calls, 41, "Number of iteration is wrong");
//     assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");

//     exp.run();

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();

//     let config = FolderConfig::new("tmp_test_fidbatchparrun_loadpool");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(
//         threaded,
//         PoolMode::Persistent,
//         BatchRandomSearch,
//         Calls,
//         sp,
//         obj,
//         (rec, check)
//     );
//     run_reader_eps("tmp_test_fidbatchparrun_loadpool", 500, 35);
//     let expstop = exp.get_stop();
//     assert!(
//         expstop >= 100 && expstop <= 107,
//         "Number of calls is wrong"
//     );
//     let expoptimizer = exp.get_optimizer();
//     assert_eq!(expoptimizer.calls, 76, "Number of iteration is wrong");
//     assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");
// }



// #[test]
// fn test_fid_thr_seq_run_loadpool() {
//     let _clean = Cleaner::new("tmp_test_fidthrseqrun_loadpool");

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();
//     let opt = RandomSearch::new();
//     let stop = Calls::new(50);
//     let config = FolderConfig::new("tmp_test_fidthrseqrun_loadpool");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded_with_pool(sp, obj, opt, stop, (rec, check), PoolMode::Persistent);
//     exp.run();
//     run_reader_eps("tmp_test_fidthrseqrun_loadpool", 250, 249);

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();

//     let config = FolderConfig::new("tmp_test_fidthrseqrun_loadpool");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         PoolMode::Persistent,
//         RandomSearch,
//         Calls,
//         sp,
//         obj,
//         (rec, check)
//     );

//     let expstop: &mut Calls = exp.get_mut_stop();
//     let max_calls = 50 + num_cpus::get();
//     assert!(
//         expstop.calls() >= 50 && expstop.calls() <= max_calls,
//         "Number of calls is wrong, should be between 50 and {}",
//         max_calls
//     );
//     expstop.add(50);

//     exp.run();

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();

//     let config = FolderConfig::new("tmp_test_fidthrseqrun_loadpool");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(
//         threaded,
//         PoolMode::Persistent,
//         RandomSearch,
//         Calls,
//         sp,
//         obj,
//         (rec, check)
//     );
//     run_reader_eps("tmp_test_fidthrseqrun_loadpool", 500, 249);
//     let expstop: &Calls = exp.get_stop();
//     let max_calls = 100 + 2 * num_cpus::get();
//     assert!(
//         expstop.calls() >= 100 && expstop.calls() <= max_calls,
//         "Number of calls is wrong, should be between 50 and {}",
//         max_calls
//     );
// }
