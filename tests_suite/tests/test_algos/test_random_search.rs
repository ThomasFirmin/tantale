use tantale::algos::{NoPruner, batchrandomsearch, nopruner};
use tantale::algos::{BatchRandomSearch, random_search::RandomSearch};
use tantale::core::Experiment;
use tantale::core::problem::Pb;
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack,
    experiment::Runable,
    stop::Evaluated,
};

use crate::cleaner::Cleaner;
use crate::init_func::{sp_evaluator, sp_evaluator_fid};
use crate::run_checker::{run_reader, run_reader_eps};

#[test]
fn test_batch_run() {
    let _clean = Cleaner::new("tmp_test_batchrun");

    let sp = sp_evaluator::get_searchspace();
    let obj = sp_evaluator::get_function();
    let pb = Pb::new(sp, obj);

    let opt = BatchRandomSearch::new(7);
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_batchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader("tmp_test_batchrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);

    let config = FolderConfig::new("tmp_test_batchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, batchrandomsearch!(), Evaluated, _, _>::load(pb, rec, check);

    let expstop: &mut Evaluated = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    let expoptimizer = exp.sampler();
    assert_eq!(expoptimizer.calls, 8, "Number of iteration is wrong");
    assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");
    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);

    let config = FolderConfig::new("tmp_test_batchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, batchrandomsearch!(), Evaluated, _, _>::load(pb, rec, check);
    let expstop: &Evaluated = exp.get_stop();
    let expoptimizer = exp.sampler();
    assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
    assert_eq!(expoptimizer.calls, 15, "Number of iteration is wrong");
    assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");

    run_reader("tmp_test_batchrun", 100);
}



#[test]
fn test_seqrun() {
    let _clean = Cleaner::new("tmp_test_seqrun");

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);

    let opt = RandomSearch::new();
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_seqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader("tmp_test_seqrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);


    let config = FolderConfig::new("tmp_test_seqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, RandomSearch, Evaluated, _, _>::load(pb, rec, check);

    let expstop: &mut Evaluated = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);


    let config = FolderConfig::new("tmp_test_seqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let _exp = Experiment::<_, RandomSearch, Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_seqrun", 100);
}

#[test]
fn test_fid_batch_run() {
    let _clean = Cleaner::new("tmp_test_fidbatchrun");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let smpl = BatchRandomSearch::new(7);
    let opt = NoPruner::new(smpl, None);

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_fidbatchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader("tmp_test_fidbatchrun", 250);

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_fidbatchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, nopruner!(batchrandomsearch!()), Evaluated, _, _>::load(pb, rec, check);

    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.sampler();
    assert_eq!(expoptimizer.sampler.calls, 8,"Number of iteration is wrong");
    assert_eq!(expoptimizer.sampler.batch_size, 7, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);
     
    let config = FolderConfig::new("tmp_test_fidbatchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, nopruner!(batchrandomsearch!()), Evaluated, _, _>::load(pb, rec, check);
    run_reader_eps("tmp_test_fidbatchrun", 500, num_cpus::get() * 4);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.sampler();
    assert_eq!(expoptimizer.sampler.calls, 15, "Number of iteration is wrong");
    assert_eq!(expoptimizer.sampler.batch_size, 7, "Batch size is wrong");
}

#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_fidseqrun");

    let pb = sp_evaluator_fid::get_problem();
    let smpl = RandomSearch::new();
    let opt = NoPruner::new(smpl, None);

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_fidseqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();
    run_reader("tmp_test_fidseqrun", 250);

    let pb = sp_evaluator_fid::get_problem();

    let config = FolderConfig::new("tmp_test_fidseqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, nopruner!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);

    let expstop: &mut Evaluated = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let pb = sp_evaluator_fid::get_problem();

    let config = FolderConfig::new("tmp_test_fidseqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, nopruner!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_fidseqrun", 500);
    let expstop: &Evaluated = exp.get_stop();
    assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
}

// #[test]
// fn test_fid_thr_seq_run() {
//     let _clean = Cleaner::new("tmp_test_fidthrseqrun");

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();
//     let opt = RandomSearch::new();
//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_fidthrseqrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();
//     run_reader_eps("tmp_test_fidthrseqrun", 250, num_cpus::get() * 4);

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();

//     let config = FolderConfig::new("tmp_test_fidthrseqrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(threaded, RandomSearch, Evaluated, sp, obj, (rec, check));

//     let expstop: &mut Evaluated = exp.mut_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 50 && expstop.calls() <= max_call,
//         "Number of calls is wrong, it should be between 50 and {}",
//         max_call
//     );
//     expstop.add(50);

//     exp.run();

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();

//     let config = FolderConfig::new("tmp_test_fidthrseqrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(threaded, RandomSearch, Evaluated, sp, obj, (rec, check));
//     run_reader_eps("tmp_test_fidthrseqrun", 500, num_cpus::get() * 4);
//     let expstop: &Evaluated = exp.get_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 100 && expstop.calls() <= max_call,
//         "Number of calls is wrong, it should be between 100 and {}",
//         max_call
//     );
// }


// #[test]
// fn test_batch_parrun() {
//     let _clean = Cleaner::new("tmp_test_parbatchrun");

//     let sp = sp_evaluator::get_searchspace();
//     let func = sp_evaluator::get_function();
//     let opt = BatchRandomSearch::new(7);
//     let pb = Pb::new(sp, func);

//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_parbatchrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     run_reader("tmp_test_parbatchrun", 50);

//     let sp = sp_evaluator::get_searchspace();
//     let func = sp_evaluator::get_function();
//     let pb = Pb::new(sp, func);


//     let config = FolderConfig::new("tmp_test_parbatchrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         BatchRandomSearch,
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );

//     let expstop: &mut Evaluated = exp.mut_stop();
//     assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
//     expstop.add(50);

//     let expoptimizer = exp.get_mut_optimizer();
//     assert_eq!(expoptimizer.calls, 9, "Number of iteration is wrong");
//     assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");

//     exp.run();

//     let sp = sp_evaluator::get_searchspace();
//     let func = sp_evaluator::get_function();
//     let pb = Pb::new(sp, func);


//     let config = FolderConfig::new("tmp_test_parbatchrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(
//         threaded,
//         BatchRandomSearch,
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );
//     run_reader("tmp_test_parbatchrun", 100);

//     let expstop: &Evaluated = exp.get_stop();
//     assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
//     let expoptimizer = exp.sampler();
//     assert_eq!(expoptimizer.calls, 17, "Number of iteration is wrong");
//     assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");
// }

// #[test]
// fn test_thrseqrun() {
//     let _clean = Cleaner::new("tmp_test_thr_seq_run");

//     let sp = sp_evaluator::get_searchspace();
//     let obj = sp_evaluator::get_function();
//     let opt = RandomSearch::new();
//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_thr_seq_run");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = ThrExperiment::new(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     run_reader_eps("tmp_test_thr_seq_run", 50, num_cpus::get() * 4);

//     let sp = sp_evaluator::get_searchspace();
//     let func = sp_evaluator::get_function();
//     let pb = Pb::new(sp, func);


//     let config = FolderConfig::new("tmp_test_thr_seq_run");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(threaded, RandomSearch, Evaluated, sp, obj, (rec, check));

//     let expstop: &mut Evaluated = exp.mut_stop();
//     let calls = expstop.calls();
//     assert!((50..=55).contains(&calls), "Number of calls is wrong");
//     expstop.add(50);

//     exp.run();

//     let sp = sp_evaluator::get_searchspace();
//     let func = sp_evaluator::get_function();
//     let pb = Pb::new(sp, func);


//     let config = FolderConfig::new("tmp_test_thr_seq_run");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(threaded, RandomSearch, Evaluated, sp, obj, (rec, check));
//     let expstop: &Evaluated = exp.get_stop();
//     let calls = expstop.calls();
//     assert!((100..=105).contains(&calls), "Number of calls is wrong");
// }

// #[test]
// fn test_fid_batch_parrun() {
//     let _clean = Cleaner::new("tmp_test_fidbatchparrun");

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();
//     let opt = BatchRandomSearch::new(7);

//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_fidbatchparrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = Experiment::new(pb, opt, stop, rec, check);
//     exp.run();

//     // 56 = 7 batches * 8 iteration. 5 partials to get to evaluated
//     // -6 as the last batch stops when one last evaluation reach Evaluated
//     run_reader("tmp_test_fidbatchparrun", 56 * 5 - 6);

//     let sp = sp_evaluator_fid::get_searchspace();
//     let obj = sp_evaluator_fid::get_function();

//     let config = FolderConfig::new("tmp_test_fidbatchparrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = Experiment::<_, batchrandomsearch!(), Evaluated, _, _>::load(pb, rec, check);

//     let expstop = exp.mut_stop();
//     assert_eq!(expstop.0, 50, "Number of calls is wrong");
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

//     let exp = Experiment::<_, batchrandomsearch!(), Evaluated, _, _>::load(pb, rec, check);
//     // times 2
//     run_reader("tmp_test_fidbatchparrun", 548);
//     let expstop = exp.get_stop();
//     assert_eq!(expstop.0, 100, "Number of calls is wrong");
//     let expoptimizer = exp.sampler();
//     assert_eq!(expoptimizer.calls, 81, "Number of iteration is wrong");
//     assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");
// }