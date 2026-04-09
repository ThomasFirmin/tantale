use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack, Objective, SaverConfig, SingleCodomain,
    experiment::{MonoExperiment, Runable, ThrExperiment, mono, threaded},
    load,
    stop::Calls,
};
use tantale::algos::{
    BatchRandomSearch,
    random_search::{self, RandomSearch},
};

use crate::init_func::{sp_evaluator, OutEvaluator};
use crate::cleaner::Cleaner;
use crate::run_checker::{run_reader, run_reader_eps};

#[test]
fn test_batch_run() {
    let _clean = Cleaner::new("tmp_test_batchrun");

    let sp = sp_evaluator::get_searchspace();
    let obj = sp_evaluator::get_function();
    let opt = BatchRandomSearch::new(7);
    let cod: SingleCodomain<OutEvaluator> = random_search::codomain(|o: &OutEvaluator| o.obj);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_batchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = MonoExperiment::new((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_batchrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_batchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, BatchRandomSearch, Calls, (sp, cod), obj, (rec, check));

    let expstop: &mut Calls = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    let expoptimizer = exp.get_mut_optimizer();
    assert_eq!(expoptimizer.0.iteration, 9, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");
    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_batchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, BatchRandomSearch, Calls, (sp, cod), obj, (rec, check));
    let expstop: &Calls = exp.get_stop();
    let expoptimizer = exp.get_optimizer();
    assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
    assert_eq!(expoptimizer.0.iteration, 17, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");

    run_reader("tmp_test_batchrun", 100);
}

#[test]
fn test_batch_parrun() {
    let _clean = Cleaner::new("tmp_test_parbatchrun");

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let opt = BatchRandomSearch::new(7);
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_parbatchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_parbatchrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_parbatchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        threaded,
        BatchRandomSearch,
        Calls,
        (sp, cod),
        obj,
        (rec, check)
    );

    let expstop: &mut Calls = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    let expoptimizer = exp.get_mut_optimizer();
    assert_eq!(expoptimizer.0.iteration, 9, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_parbatchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(
        threaded,
        BatchRandomSearch,
        Calls,
        (sp, cod),
        obj,
        (rec, check)
    );
    run_reader("tmp_test_parbatchrun", 100);

    let expstop: &Calls = exp.get_stop();
    assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(expoptimizer.0.iteration, 17, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");
}

#[test]
fn test_seqrun() {
    let _clean = Cleaner::new("tmp_test_seqrun");

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let opt = RandomSearch::new();
    let cod: SingleCodomain<OutEvaluator> = random_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_seqrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, RandomSearch, Calls, (sp, cod), obj, (rec, check));

    let expstop: &mut Calls = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let _exp = load!(mono, RandomSearch, Calls, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_seqrun", 100);
}

#[test]
fn test_thrseqrun() {
    let _clean = Cleaner::new("tmp_test_thr_seq_run");

    let sp = sp_evaluator::get_searchspace();
    let obj = sp_evaluator::get_function();
    let opt = RandomSearch::new();
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_thr_seq_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = ThrExperiment::new((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader_eps("tmp_test_thr_seq_run", 50, 5);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_thr_seq_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(threaded, RandomSearch, Calls, (sp, cod), obj, (rec, check));

    let expstop: &mut Calls = exp.get_mut_stop();
    let calls = expstop.calls();
    assert!((50..=55).contains(&calls), "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_thr_seq_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(threaded, RandomSearch, Calls, (sp, cod), obj, (rec, check));
    let expstop: &Calls = exp.get_stop();
    let calls = expstop.calls();
    assert!((100..=105).contains(&calls), "Number of calls is wrong");
}
