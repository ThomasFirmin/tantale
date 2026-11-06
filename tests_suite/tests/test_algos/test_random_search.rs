use tantale::algos::{NoPruner, batchrandomsearch, nopruner};
use tantale::algos::{BatchRandomSearch, random_search::RandomSearch};
use tantale::core::Experiment;
use tantale::core::ParExperiment;
use tantale::core::experiment::parallel::threaded::ThreadCommunicator;
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

    let expstop: &mut Evaluated = &mut exp.stop;
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    let expoptimizer = &exp.sampler;
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
    let expstop: &Evaluated = &exp.stop;
    let expoptimizer = exp.sampler;
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

    let expstop: &mut Evaluated = &mut exp.stop;
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

    let expstop = &mut exp.stop;
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = &exp.sampler;
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
    let expstop = &exp.stop;
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.sampler;
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

    let expstop: &mut Evaluated = &mut exp.stop;
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let pb = sp_evaluator_fid::get_problem();

    let config = FolderConfig::new("tmp_test_fidseqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, nopruner!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_fidseqrun", 500);
    let expstop: &Evaluated = &exp.stop;
    assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
}

#[test]
fn test_batch_parrun() {
    let _clean = Cleaner::new("tmp_test_batch_parrun");

    let sp = sp_evaluator::get_searchspace();
    let obj = sp_evaluator::get_function();
    let pb = Pb::new(sp, obj);

    let opt = BatchRandomSearch::new(7);
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_batch_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let com = ThreadCommunicator::new();
    let exp = ParExperiment::new(pb, opt, stop, rec, check, com);
    exp.run();

    run_reader("tmp_test_batch_parrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);

    let config = FolderConfig::new("tmp_test_batch_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let mut exp = ParExperiment::<_, batchrandomsearch!(), Evaluated, _, _, _>::load(pb, rec, check, com);

    if let Some(m) = exp.master_mut() {
        let expstop: &mut Evaluated = &mut m.stop;
        assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
        expstop.add(50);
        let expoptimizer = &m.sampler;
        assert!(expoptimizer.calls >=8, "Number of iteration is wrong. Should be at least 8, but got {}", expoptimizer.calls);
        assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");
    }
    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);

    let config = FolderConfig::new("tmp_test_batch_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let mut exp = ParExperiment::<_, batchrandomsearch!(), Evaluated, _, _, _>::load(pb, rec, check, com);
    if let Some(m) = exp.master_mut() {
        let expstop: &Evaluated = &m.stop;
        let expoptimizer = &m.sampler;
        assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
        assert!(expoptimizer.calls >=15, "Number of iteration is wrong. Should be at least 15, but got {}", expoptimizer.calls);
        assert_eq!(expoptimizer.batch_size, 7, "Batch size is wrong");
    }

    run_reader("tmp_test_batch_parrun", 100);
}

#[test]
fn test_parrun() {
    let _clean = Cleaner::new("tmp_test_parrun");

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);

    let opt = RandomSearch::new();
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);


    let com = ThreadCommunicator::new();
    let exp = ParExperiment::new(pb, opt, stop, rec, check, com);
    exp.run();

    run_reader("tmp_test_parrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);


    let config = FolderConfig::new("tmp_test_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let mut exp = ParExperiment::<_, RandomSearch, Evaluated, _, _, _>::load(pb, rec, check, com);
    if let Some(m) = exp.master_mut() {
        let expstop: &mut Evaluated = &mut m.stop;
        assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
        expstop.add(50);
    }
    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::get_function();
    let pb = Pb::new(sp, func);


    let config = FolderConfig::new("tmp_test_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let _exp = ParExperiment::<_, RandomSearch, Evaluated, _, _, _>::load(pb, rec, check, com);

    run_reader("tmp_test_parrun", 100);
}