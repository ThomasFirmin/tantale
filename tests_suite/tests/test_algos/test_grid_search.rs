use tantale::algos::{GridSearch, NoPruner, nopruner};
use tantale::core::ParExperiment;
use tantale::core::experiment::parallel::threaded::ThreadCommunicator;
use tantale::core::{Experiment, Problem};
use tantale::core::problem::Pb;
use tantale::{
    core::{
        CSVRecorder, FolderConfig, MessagePack,
        experiment::Runable,
        stop::Evaluated,
    },
};

use crate::cleaner::Cleaner;
use crate::init_func::{sp_grid_evaluator, sp_grid_evaluator_fid};
use crate::run_checker::run_reader;

#[test]
fn test_seqrun() {
    let _clean = Cleaner::new("tmp_test_gs_seqrun");

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let pb = Pb::new(sp, obj);

    let opt = GridSearch::new(pb.get_searchspace());

    let stop = Evaluated::new(200);
    let config = FolderConfig::new("tmp_test_gs_seqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader("tmp_test_gs_seqrun", 200);

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_gs_seqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, GridSearch, Evaluated, _, _>::load(pb, rec, check);
    assert_eq!(exp.sampler.1, 0, "Number of fully evaluated grid is wrong");

    let expstop: &mut Evaluated = &mut exp.stop;
    assert_eq!(expstop.calls(), 200, "Number of calls is wrong");
    expstop.add(100);

    exp.run();

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_gs_seqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, GridSearch, Evaluated, _, _>::load(pb, rec, check);
    assert_eq!(exp.sampler.1, 1, "Number of fully evaluated grid is wrong");
    run_reader("tmp_test_gs_seqrun", 300);
    exp.run();
}

#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_gs_fidseqrun");

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let smpl = GridSearch::new(&sp);
    let opt = NoPruner::new(smpl, None);

    let stop = Evaluated::new(200);
    let config = FolderConfig::new("tmp_test_gs_fidseqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(Pb::new(sp, obj), opt, stop, rec, check);
    exp.run();
    run_reader("tmp_test_gs_fidseqrun", 1000);

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();

    let config = FolderConfig::new("tmp_test_gs_fidseqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, nopruner!(GridSearch), Evaluated, _, _>::load(Pb::new(sp, obj), rec, check);
    assert_eq!(exp.sampler.sampler.1, 0, "Number of fully evaluated grid is wrong");

    let expstop: &mut Evaluated = &mut exp.stop;
    assert_eq!(expstop.calls(), 200, "Number of calls is wrong");
    expstop.add(100);

    exp.run();

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();

    let config = FolderConfig::new("tmp_test_gs_fidseqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();


    let exp = Experiment::<_, nopruner!(GridSearch), Evaluated, _, _>::load(Pb::new(sp, obj), rec, check);
    assert_eq!(exp.sampler.sampler.1, 1, "Number of fully evaluated grid is wrong");
    run_reader("tmp_test_gs_fidseqrun", 1500);
    let expstop: &Evaluated = &exp.stop;
    assert_eq!(expstop.calls(), 300, "Number of calls is wrong");
}

#[test]
fn test_seq_parrun() {
    let _clean = Cleaner::new("tmp_test_gs_seq_parrun");

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let pb = Pb::new(sp, obj);

    let opt = GridSearch::new(pb.get_searchspace());

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_gs_seq_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let com = ThreadCommunicator::new();
    let exp = ParExperiment::new(pb, opt, stop, rec, check, com);
    exp.run();

    run_reader("tmp_test_gs_seq_parrun", 50);

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_gs_seq_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let mut exp = ParExperiment::<_, GridSearch, Evaluated, _, _, _>::load(pb, rec, check, com);
    if let Some(m) = exp.master_mut() {
        let expstop: &mut Evaluated = &mut m.stop;
        assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
        expstop.add(50);
    }
    exp.run();

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_gs_seq_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let mut _exp = ParExperiment::<_, GridSearch, Evaluated, _, _, _>::load(pb, rec, check, com);
    match _exp.master_mut() {
        Some(m) => {
            let expstop= &m.stop;
            assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
        },
        None => panic!("Master process not found"),
    }
    run_reader("tmp_test_gs_seq_parrun", 100);
}