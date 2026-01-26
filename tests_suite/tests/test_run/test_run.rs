use tantale_core::{
    CSVRecorder, FolderConfig, MessagePack, Objective, SaverConfig, SingleCodomain, experiment::{MonoExperiment, Runable, ThrExperiment, mono, threaded}, load, solution::shape::RawObj, stop::Calls
};

use tantale_algos::{BatchRandomSearch, random_search::RandomSearch};

use super::init_func::sp_evaluator;
use crate::init_func::OutEvaluator;

use std::path::Path;

struct Cleaner {
    path: String,
}

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

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

#[test]
fn test_batch_run() {
    drop(Cleaner {
        path: String::from("tmp_test_batchrun"),
    });
    let _clean = Cleaner {
        path: String::from("tmp_test_batchrun"),
    };

    let sp = sp_evaluator::get_searchspace();
    let obj = sp_evaluator::get_function();
    let opt = BatchRandomSearch::new(7);
    let cod: SingleCodomain<OutEvaluator> = BatchRandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_batchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1);

    let exp = MonoExperiment::new((sp,cod), obj, opt, stop, (rec,check));
    exp.run();

    run_reader("tmp_test_batchrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = BatchRandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_batchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let mut exp = load!(mono, BatchRandomSearch, Calls, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    
    let expoptimizer = exp.get_mut_optimizer();
    assert_eq!(expoptimizer.0.iteration, 9, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = BatchRandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_batchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let exp = load!(mono, BatchRandomSearch, Calls, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_batchrun", 100);
    let expstop = exp.get_stop();
    let expoptimizer = exp.get_optimizer();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    assert_eq!(expoptimizer.0.iteration, 17,"Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");

    drop(Cleaner {
        path: String::from("tmp_test_batchrun"),
    });
}

#[test]
fn test_batch_parrun() {
    drop(Cleaner {
        path: String::from("tmp_test_parbatchrun"),
    });
    let _clean = Cleaner {
        path: String::from("tmp_test_parbatchrun"),
    };

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let opt = BatchRandomSearch::new(7);
    let cod = BatchRandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_parbatchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1);

    let exp = threaded((sp,cod), obj, opt, stop, (rec,check));
    exp.run();

    run_reader("tmp_test_parbatchrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = BatchRandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_parbatchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let mut exp = load!(threaded, BatchRandomSearch, Calls, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;

    let expoptimizer = exp.get_mut_optimizer();
    assert_eq!(expoptimizer.0.iteration, 9, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = BatchRandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_parbatchrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let exp = load!(threaded, BatchRandomSearch, Calls, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_parbatchrun", 100);

    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.iteration, 17,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.0.batch, 7, "Batch size is wrong");
}









#[test]
fn test_seqrun() {
    drop(Cleaner {
        path: String::from("tmp_test_seqrun"),
    });
    let _clean = Cleaner {
        path: String::from("tmp_test_seqrun"),
    };

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let opt = RandomSearch::new();
    let cod: SingleCodomain<OutEvaluator> = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1);

    let exp = mono((sp,cod), obj, opt, stop, (rec,check));
    exp.run();

    run_reader("tmp_test_seqrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let mut exp = load!(mono, RandomSearch, Calls, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;

    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let exp = load!(mono, RandomSearch, Calls, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_seqrun", 100);
    assert_eq!(exp.get_stop().0, 100, "Number of calls is wrong");
    drop(Cleaner {
        path: String::from("tmp_test_seqrun"),
    });
}







#[test]
fn test_thrseqrun() {
    drop(Cleaner {
        path: String::from("tmp_test_thr_seq_run"),
    });
    let _clean = Cleaner {
        path: String::from("tmp_test_thr_seq_run"),
    };

    let sp = sp_evaluator::get_searchspace();
    let obj = sp_evaluator::get_function();
    let opt = RandomSearch::new();
    let cod = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_thr_seq_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1);

    let exp = ThrExperiment::new((sp,cod),obj,opt,stop,(rec,check));
    exp.run();

    run_reader("tmp_test_thr_seq_run", 50);
}