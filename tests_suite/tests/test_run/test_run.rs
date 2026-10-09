use tantale::algos::batchrandomsearch;
use tantale::algos::{BatchRandomSearch, random_search::RandomSearch};
use tantale::core::problem::Pb;
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack,
    experiment::{Experiment, Runable},
    stop::Calls,
};

use crate::cleaner::Cleaner;
use crate::init_func::sp_evaluator;
use crate::run_checker::{run_reader, run_reader_eps};

#[test]
fn test_batch_run() {
    let _clean = Cleaner::new("tmp_test_batchrun");

    let sp = sp_evaluator::get_searchspace();
    let obj = sp_evaluator::get_function();
    let pb = Pb::new(sp, obj);

    let opt = BatchRandomSearch::new(7);

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_batchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader("tmp_test_batchrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let obj = sp_evaluator::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_batchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, batchrandomsearch!(),Calls,_,_>::load(pb, rec, check);

    let expstop: &mut Calls = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    let expoptimizer = exp.mut_sampler();
    assert_eq!(expoptimizer.calls, 9, "Number of iteration is wrong");
    assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");
    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);
    

    let config = FolderConfig::new("tmp_test_batchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, batchrandomsearch!(),Calls,_,_>::load(pb, rec, check);
    let expstop: &Calls = exp.get_stop();
    let expoptimizer = exp.sampler();
    assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
    assert_eq!(expoptimizer.calls, 17, "Number of iteration is wrong");
    assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");

    run_reader("tmp_test_batchrun", 100);
}

#[test]
fn test_seqrun() {
    let _clean = Cleaner::new("tmp_test_seqrun");

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_seqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, RandomSearch::new(), stop, rec, check);
    exp.run();

    run_reader("tmp_test_seqrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);

    let config = FolderConfig::new("tmp_test_seqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, RandomSearch, Calls, _, _>::load(pb, rec, check);

    let expstop: &mut Calls = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);

    let config = FolderConfig::new("tmp_test_seqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let _exp = Experiment::<_, RandomSearch, Calls, _, _>::load(pb, rec, check);
    run_reader("tmp_test_seqrun", 100);
}

// #[test]
// fn test_batch_parrun() {
//     let _clean = Cleaner::new("tmp_test_parbatchrun");

//     let sp = sp_evaluator::get_searchspace();
//     let func = sp_evaluator::example;
//     let opt = BatchRandomSearch::new(7);
//     let obj = OneStep::new(func);
//     let stop = Calls::new(50);
//     let config = FolderConfig::new("tmp_test_parbatchrun").init();
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     run_reader("tmp_test_parbatchrun", 50);

//     let sp = sp_evaluator::get_searchspace();
//     let func = sp_evaluator::example;
//     let obj = OneStep::new(func);

//     let config = FolderConfig::new("tmp_test_parbatchrun").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(threaded, BatchRandomSearch, Calls, sp, obj, (rec, check));

//     let expstop: &mut Calls = exp.get_mut_stop();
//     assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
//     expstop.add(50);

//     let expoptimizer = exp.get_mut_optimizer();
//     assert_eq!(expoptimizer.0.iteration, 9, "Number of iteration is wrong");
//     assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");

//     exp.run();

//     let sp = sp_evaluator::get_searchspace();
//     let func = sp_evaluator::example;
//     let obj = OneStep::new(func);

//     let config = FolderConfig::new("tmp_test_parbatchrun").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(threaded, BatchRandomSearch, Calls, sp, obj, (rec, check));
//     run_reader("tmp_test_parbatchrun", 100);

//     let expstop: &Calls = exp.get_stop();
//     assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
//     let expoptimizer = exp.get_optimizer();
//     assert_eq!(expoptimizer.0.iteration, 17, "Number of iteration is wrong");
//     assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");
// }

// #[test]
// fn test_thrseqrun() {
//     let _clean = Cleaner::new("tmp_test_thr_seq_run");

//     let sp = sp_evaluator::get_searchspace();
//     let obj = sp_evaluator::get_function();
//     let opt = RandomSearch::new();
//     let stop = Calls::new(50);
//     let config = FolderConfig::new("tmp_test_thr_seq_run").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config);

//     let exp = ThrExperiment::new(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     run_reader_eps("tmp_test_thr_seq_run", 50, 5);

//     let sp = sp_evaluator::get_searchspace();
//     let func = sp_evaluator::example;
//     let obj = OneStep::new(func);

//     let config = FolderConfig::new("tmp_test_thr_seq_run").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(threaded, RandomSearch, Calls, sp, obj, (rec, check));

//     let expstop: &mut Calls = exp.get_mut_stop();
//     let calls = expstop.calls();
//     assert!((50..=55).contains(&calls), "Number of calls is wrong");
//     expstop.add(50);

//     exp.run();

//     let sp = sp_evaluator::get_searchspace();
//     let func = sp_evaluator::example;
//     let obj = OneStep::new(func);

//     let config = FolderConfig::new("tmp_test_thr_seq_run").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(threaded, RandomSearch, Calls, sp, obj, (rec, check));
//     let expstop: &Calls = exp.get_stop();
//     let calls = expstop.calls();
//     assert!((100..=105).contains(&calls), "Number of calls is wrong");
// }
