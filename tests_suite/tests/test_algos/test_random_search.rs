use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack, Objective, SaverConfig, SingleCodomain,
    experiment::{MonoExperiment, ThrExperiment},
    experiment::{Runable, mono, threaded},
    load,
    stop::Calls,
};
use tantale::algos::{
    BatchRandomSearch,
    random_search::{self, RandomSearch},
};

use crate::init_func::{sp_evaluator,OutEvaluator,sp_evaluator_fid,FidOutEvaluator};
use crate::cleaner::Cleaner;
use std::path::Path;

pub fn run_reader(path: &str, size: usize) {
    let true_path = Path::new(path);
    let eval_path = true_path.join(Path::new("recorder"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");
    let path_info = eval_path.join("info.csv");

    // Check `Obj`, `Opt`, `Codom`
    let mut rdr_obj = csv::Reader::from_path(path_obj).unwrap();
    let mut rdr_opt = csv::Reader::from_path(path_opt).unwrap();
    let mut rdr_cod = csv::Reader::from_path(path_cod).unwrap();
    let mut rdr_info = csv::Reader::from_path(path_info).unwrap();
    let mut rdr_out = csv::Reader::from_path(path_out).unwrap();

    let linesobj = rdr_obj.records();
    let linesopt = rdr_opt.records();
    let linescod = rdr_cod.records();
    let linesinfo = rdr_info.records();
    let linesout = rdr_out.records();

    let count_obj = linesobj.count();
    let count_opt = linesopt.count();
    let count_cod = linescod.count();
    let count_info = linesinfo.count();
    let count_out = linesout.count();

    let linesobj = rdr_obj.records();
    linesobj.for_each(|l| println!("{:?}", l));
    assert_eq!(count_obj, size, "Some solutions are missing in obj.");
    assert_eq!(count_opt, size, "Some solutions are missing in opt.");
    assert_eq!(count_cod, size, "Some solutions are missing in cod.");
    assert_eq!(count_info, size, "Some solutions are missing in info.");
    assert_eq!(count_out, size, "Some solutions are missing in out.");
}

pub fn run_reader_eps(path: &str, size: usize, epsilon: usize) {
    let true_path = Path::new(path);
    let eval_path = true_path.join(Path::new("recorder"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");
    let path_info = eval_path.join("info.csv");

    // Check `Obj`, `Opt`, `Codom`
    let mut rdr_obj = csv::Reader::from_path(path_obj).unwrap();
    let mut rdr_opt = csv::Reader::from_path(path_opt).unwrap();
    let mut rdr_cod = csv::Reader::from_path(path_cod).unwrap();
    let mut rdr_info = csv::Reader::from_path(path_info).unwrap();
    let mut rdr_out = csv::Reader::from_path(path_out).unwrap();

    let linesobj = rdr_obj.records();
    let linesopt = rdr_opt.records();
    let linescod = rdr_cod.records();
    let linesinfo = rdr_info.records();
    let linesout = rdr_out.records();

    let count_obj = linesobj.count();
    let count_opt = linesopt.count();
    let count_cod = linescod.count();
    let count_info = linesinfo.count();
    let count_out = linesout.count();

    let linesobj = rdr_obj.records();
    linesobj.for_each(|l| println!("{:?}", l));
    assert!(
        (count_obj >= size) && (count_obj < size + epsilon),
        "Some solutions are missing in obj."
    );
    assert!(
        (count_opt >= size) && (count_opt < size + epsilon),
        "Some solutions are missing in opt."
    );
    assert!(
        (count_cod >= size) && (count_cod < size + epsilon),
        "Some solutions are missing in cod."
    );
    assert!(
        (count_info >= size) && (count_info < size + epsilon),
        "Some solutions are missing in info."
    );
    assert!(
        (count_out >= size) && (count_out < size + epsilon),
        "Some solutions are missing in out."
    );
    assert!(
        [count_opt, count_cod, count_info, count_out]
            .iter()
            .all(|c| c == &count_obj),
        "Not all counts are equal. Some solutions are missing within at least one save file"
    );
}

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

#[test]
fn test_fid_batch_run() {
    let _clean = Cleaner::new("tmp_test_fidbatchrun");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let opt = BatchRandomSearch::new(7);
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_fidbatchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader_eps("tmp_test_fidbatchrun", 264, 20);

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_fidbatchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, BatchRandomSearch, Calls, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.get_optimizer();
    assert!(
        (expoptimizer.0.iteration >= 39) && (expoptimizer.0.iteration <= 41),
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_fidbatchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, BatchRandomSearch, Calls, (sp, cod), obj, (rec, check));
    run_reader_eps("tmp_test_fidbatchrun", 524, 40);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(expoptimizer.0.iteration, 81, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");
}

#[test]
fn test_fid_batch_parrun() {
    let _clean = Cleaner::new("tmp_test_fidbatchparrun");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let opt = BatchRandomSearch::new(7);
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_fidbatchparrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader_eps("tmp_test_fidbatchparrun", 274, 20);

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_fidbatchparrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, BatchRandomSearch, Calls, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.get_mut_optimizer();
    assert_eq!(expoptimizer.0.iteration, 41, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_fidbatchparrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, BatchRandomSearch, Calls, (sp, cod), obj, (rec, check));
    run_reader_eps("tmp_test_fidbatchparrun", 548, 20);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(expoptimizer.0.iteration, 81, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");
}

#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_fidseqrun");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let opt = RandomSearch::new();
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_fidseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();
    run_reader("tmp_test_fidseqrun", 250);

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_fidseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, RandomSearch, Calls, (sp, cod), obj, (rec, check));

    let expstop: &mut Calls = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_fidseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, RandomSearch, Calls, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_fidseqrun", 500);
    let expstop: &Calls = exp.get_stop();
    assert_eq!(expstop.calls(), 100, "Number of calls is wrong");
}

#[test]
fn test_fid_thr_seq_run() {
    let _clean = Cleaner::new("tmp_test_fidthrseqrun");

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let opt = RandomSearch::new();
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_fidthrseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    exp.run();
    run_reader_eps("tmp_test_fidthrseqrun", 250, 249);

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_fidthrseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(threaded, RandomSearch, Calls, (sp, cod), obj, (rec, check));

    let expstop: &mut Calls = exp.get_mut_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(
        expstop.calls() >= 50 && expstop.calls() <= max_call,
        "Number of calls is wrong, it should be between 50 and {}",
        max_call
    );
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let obj = sp_evaluator_fid::get_function();
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_fidthrseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(threaded, RandomSearch, Calls, (sp, cod), obj, (rec, check));
    run_reader_eps("tmp_test_fidthrseqrun", 500, 249);
    let expstop: &Calls = exp.get_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(
        expstop.calls() >= 100 && expstop.calls() <= max_call,
        "Number of calls is wrong, it should be between 100 and {}",
        max_call
    );
}
