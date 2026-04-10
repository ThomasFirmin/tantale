use tantale::algos::{GridSearch, grid_search};
use tantale::{
    algos::grid_search::GSState,
    core::{
        CSVRecorder, FolderConfig, MessagePack, Objective, SaverConfig, SingleCodomain,
        experiment::{Runable, ThrExperiment, mono, threaded},
        load,
        stop::Evaluated,
    },
};

use crate::cleaner::Cleaner;
use crate::init_func::{FidOutEvaluator, OutEvaluator, sp_grid_evaluator, sp_grid_evaluator_fid};
use crate::run_checker::{run_reader, run_reader_eps};

#[test]
fn test_seqrun() {
    let _clean = Cleaner::new("tmp_test_gs_seqrun");

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let opt = GridSearch::new(&sp);
    let cod: SingleCodomain<OutEvaluator> = grid_search::codomain(|o: &OutEvaluator| o.obj);

    let stop = Evaluated::new(200);
    let config = FolderConfig::new("tmp_test_gs_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_gs_seqrun", 200);

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let cod = grid_search::codomain(|o: &OutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_gs_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, GridSearch, Evaluated, (sp, cod), obj, (rec, check));
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 0, "Number of fully evaluated grid is wrong");

    let expstop: &mut Evaluated = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 200, "Number of calls is wrong");
    expstop.add(100);

    exp.run();

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let cod = grid_search::codomain(|o: &OutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_gs_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, GridSearch, Evaluated, (sp, cod), obj, (rec, check));
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 1, "Number of fully evaluated grid is wrong");
    run_reader("tmp_test_gs_seqrun", 300);
    exp.run();
}

#[test]
fn test_thrseqrun() {
    let _clean = Cleaner::new("tmp_test_gs_thr_seq_run");

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let opt = GridSearch::new(&sp);
    let cod = grid_search::codomain(|o: &OutEvaluator| o.obj);
    let stop = Evaluated::new(200);
    let config = FolderConfig::new("tmp_test_gs_thr_seq_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = ThrExperiment::new((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader_eps("tmp_test_gs_thr_seq_run", 200, 5);

    let sp = sp_grid_evaluator::get_searchspace();
    let func = sp_grid_evaluator::example;
    let cod = grid_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_gs_thr_seq_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        threaded,
        GridSearch,
        Evaluated,
        (sp, cod),
        obj,
        (rec, check)
    );
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 0, "Number of fully evaluated grid is wrong");

    let expstop: &mut Evaluated = exp.get_mut_stop();
    let calls = expstop.calls();
    assert!((200..=205).contains(&calls), "Number of calls is wrong");
    expstop.add(100);

    exp.run();

    let sp = sp_grid_evaluator::get_searchspace();
    let func = sp_grid_evaluator::example;
    let cod = grid_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_gs_thr_seq_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(
        threaded,
        GridSearch,
        Evaluated,
        (sp, cod),
        obj,
        (rec, check)
    );
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 1, "Number of fully evaluated grid is wrong");
    let expstop: &Evaluated = exp.get_stop();
    let calls = expstop.calls();
    assert!((300..=305).contains(&calls), "Number of calls is wrong");
}

#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_gs_fidseqrun");

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let opt = GridSearch::new(&sp);
    let cod = grid_search::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Evaluated::new(200);
    let config = FolderConfig::new("tmp_test_gs_fidseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();
    run_reader("tmp_test_gs_fidseqrun", 1000);

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let cod = grid_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_gs_fidseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, GridSearch, Evaluated, (sp, cod), obj, (rec, check));
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 0, "Number of fully evaluated grid is wrong");

    let expstop: &mut Evaluated = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 200, "Number of calls is wrong");
    expstop.add(100);

    exp.run();

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let cod = grid_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_gs_fidseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, GridSearch, Evaluated, (sp, cod), obj, (rec, check));
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 1, "Number of fully evaluated grid is wrong");
    run_reader("tmp_test_gs_fidseqrun", 1500);
    let expstop: &Evaluated = exp.get_stop();
    assert_eq!(expstop.calls(), 300, "Number of calls is wrong");
}

#[test]
fn test_fid_thr_seq_run() {
    let _clean = Cleaner::new("tmp_test_gs_fidthrseqrun");

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let opt = GridSearch::new(&sp);
    let cod = grid_search::codomain(|o: &FidOutEvaluator| o.obj);
    let stop = Evaluated::new(200);
    let config = FolderConfig::new("tmp_test_gs_fidthrseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    exp.run();
    run_reader_eps("tmp_test_gs_fidthrseqrun", 1000, 495);

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let cod = grid_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_gs_fidthrseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        threaded,
        GridSearch,
        Evaluated,
        (sp, cod),
        obj,
        (rec, check)
    );
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 0, "Number of fully evaluated grid is wrong");

    let expstop: &mut Evaluated = exp.get_mut_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(
        expstop.calls() >= 200 && expstop.calls() <= max_call,
        "Number of calls is wrong, it should be between 200 and {}",
        max_call
    );
    expstop.add(100);

    exp.run();

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let cod = grid_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_gs_fidthrseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(
        threaded,
        GridSearch,
        Evaluated,
        (sp, cod),
        obj,
        (rec, check)
    );
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 1, "Number of fully evaluated grid is wrong");
    run_reader_eps("tmp_test_gs_fidthrseqrun", 1500, 740);
    let expstop: &Evaluated = exp.get_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(
        expstop.calls() >= 300 && expstop.calls() <= max_call,
        "Number of calls is wrong, it should be between 300 and {}",
        max_call
    );
}
