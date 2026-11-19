use tantale::algos::{BatchRandomSearch, random_search::RandomSearch};
use tantale::algos::{NoPruner, batchrandomsearch, nopruner};
use tantale::core::Experiment;
use tantale::core::problem::Pb;
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack,
    experiment::{PoolMode, Runable},
    stop::Evaluated,
};

use crate::cleaner::Cleaner;
use crate::init_func::sp_evaluator_fid;
use crate::run_checker::run_reader;

#[test]
fn test_fid_batch_run() {
    let _clean = Cleaner::new("tmp_test_fidbatchrun");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let smpl: BatchRandomSearch<_> = BatchRandomSearch::new(7);
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

    let mut exp =
        Experiment::<_, nopruner!(batchrandomsearch!()), Evaluated, _, _>::load(pb, rec, check);

    let expstop = &mut exp.stop;
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = &mut exp.sampler;
    assert_eq!(
        expoptimizer.sampler.calls, 8,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.sampler.batch_size, 7, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_fidbatchrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp =
        Experiment::<_, nopruner!(batchrandomsearch!()), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_fidbatchrun", 500);
    let expstop = &exp.stop;
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = &exp.sampler;
    assert_eq!(
        expoptimizer.sampler.calls, 15,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.sampler.batch_size, 7, "Batch size is wrong");
}

#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_fidseqrun");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let smpl = RandomSearch::new();
    let opt = NoPruner::new(smpl, None);

    let stop = Evaluated::new(50);
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

    let mut exp = Experiment::<_, nopruner!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);

    let expstop = &mut exp.stop;
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_fidseqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, nopruner!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_fidseqrun", 500);
    let expstop = &exp.stop;
    assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
}

#[test]
fn test_fid_seq_run_loadpool() {
    let _clean = Cleaner::new("tmp_test_fidseqrun_loadpool");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let smpl = RandomSearch::new();
    let opt = NoPruner::new(smpl, None);

    let stop = Evaluated::new(50);
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

    let mut exp = Experiment::<_, nopruner!(RandomSearch), Evaluated, _, _>::load(pb, rec, check)
        .with_pool(PoolMode::Persistent);

    let expstop = &mut exp.stop;
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_fidseqrun_loadpool");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, nopruner!(RandomSearch), Evaluated, _, _>::load(pb, rec, check)
        .with_pool(PoolMode::Persistent);
    run_reader("tmp_test_fidseqrun_loadpool", 500);
    let expstop = &exp.stop;
    assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
}

#[test]
fn test_fid_batch_run_loadpool() {
    let _clean = Cleaner::new("tmp_test_fidbatchrun_loadpool");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let smpl: BatchRandomSearch<_> = BatchRandomSearch::new(7);
    let opt = NoPruner::new(smpl, None);

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_fidbatchrun_loadpool");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check).with_pool(PoolMode::Persistent);
    exp.run();

    run_reader("tmp_test_fidbatchrun_loadpool", 250);

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_fidbatchrun_loadpool");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp =
        Experiment::<_, nopruner!(batchrandomsearch!()), Evaluated, _, _>::load(pb, rec, check)
            .with_pool(PoolMode::Persistent);

    let expstop = &mut exp.stop;
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = &mut exp.sampler;
    assert_eq!(
        expoptimizer.sampler.calls, 8,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.sampler.batch_size, 7, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_fidbatchrun_loadpool");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp =
        Experiment::<_, nopruner!(batchrandomsearch!()), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_fidbatchrun_loadpool", 500);
    let expstop = &exp.stop;
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = &exp.sampler;
    assert_eq!(
        expoptimizer.sampler.calls, 15,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.sampler.batch_size, 7, "Batch size is wrong");
}
