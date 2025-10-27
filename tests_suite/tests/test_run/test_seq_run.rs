use tantale_core::{
    experiment, experiment::Runable, load, saver::CSVSaver, stop::Calls, Objective,
};

use tantale_algos::RandomSearch;

use super::init_func::sp_evaluator;
use crate::init_func::OutEvaluator;

use std::{collections::HashSet, path::Path};

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
    let eval_path = true_path.join(Path::new("evaluations"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");
    let path_info = eval_path.join("info.csv");

    let mut hash_id = HashSet::new();
    let mut size_obj = 0;
    let mut size_opt = 0;
    let mut size_cod = 0;
    let mut size_out = 0;
    let mut size_info = 0;

    // Check `Obj`
    let mut rdr = csv::Reader::from_path(path_obj).unwrap();
    for l in rdr.records() {
        size_obj += 1;
        let record = l.unwrap();
        let id: usize = record[0].parse().unwrap();
        hash_id.insert(id);
    }
    // Check `Opt`
    let mut rdr = csv::Reader::from_path(path_opt).unwrap();
    for l in rdr.records() {
        size_opt += 1;
        let record = l.unwrap();
        let id: usize = record[0].parse().unwrap();
        hash_id.insert(id);
    }
    // Check `Out`
    let mut rdr = csv::Reader::from_path(path_out).unwrap();
    for l in rdr.records() {
        size_cod += 1;
        let record = l.unwrap();
        let id: usize = record[0].parse().unwrap();
        hash_id.insert(id);
    }
    // Check `Cod`
    let mut rdr = csv::Reader::from_path(path_cod).unwrap();
    for l in rdr.records() {
        size_out += 1;
        let record = l.unwrap();
        let id: usize = record[0].parse().unwrap();
        hash_id.insert(id);
    }

    // Check `Info`
    let mut rdr = csv::Reader::from_path(path_info).unwrap();
    for l in rdr.records() {
        size_info += 1;
        let record = l.unwrap();
        let id: usize = record[0].parse().unwrap();
        hash_id.insert(id);
    }

    assert_eq!(
        size_obj, size,
        "Some solutions are missing within recorded obj save."
    );
    assert_eq!(
        size_opt, size,
        "Some solutions are missing within recorded opt save."
    );
    assert_eq!(
        size_cod, size,
        "Some solutions are missing within recorded cod save."
    );
    assert_eq!(
        size_out, size,
        "Some solutions are missing within recorded out save."
    );
    assert_eq!(
        size_info, size,
        "Some solutions are missing within recorded info save."
    );
    assert_eq!(hash_id.len(), size, "Some IDs are duplicated.");
}

#[test]
fn test_seq_run() {
    drop(Cleaner {
        path: String::from("tmp_test_seqrun"),
    });

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let opt = RandomSearch::new(7);
    let cod = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(cod, func);
    let stop = Calls::new(50);
    let saver = CSVSaver::new("tmp_test_seqrun", true, true, true, true, 1);

    let exp = experiment!(Mono, RandomSearch | sp, obj, opt, stop, saver);
    exp.run();

    run_reader("tmp_test_seqrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(cod, func);
    let saver = CSVSaver::new("tmp_test_seqrun", true, true, true, true, 1);
    let mut exp = load!(Mono, RandomSearch, Calls | sp, obj, saver);

    assert_eq!(exp.stop.0, 50, "Number of calls is wrong");
    assert_eq!(exp.optimizer.0.iteration, 8, "Number of iteration is wrong");
    assert_eq!(exp.optimizer.0.batch, 7, "Batch size is wrong");

    exp.stop.1 = 100;
    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(cod, func);
    let saver = CSVSaver::new("tmp_test_seqrun", true, true, true, true, 1);
    let exp = load!(Mono, RandomSearch, Calls | sp, obj, saver);
    run_reader("tmp_test_seqrun", 100);
    assert_eq!(exp.stop.0, 100, "Number of calls is wrong");
    assert_eq!(
        exp.optimizer.0.iteration, 15,
        "Number of iteration is wrong"
    );
    assert_eq!(exp.optimizer.0.batch, 7, "Batch size is wrong");

    drop(Cleaner {
        path: String::from("tmp_test_seqrun"),
    });
}

#[test]
fn test_seq_parrun() {
    drop(Cleaner {
        path: String::from("tmp_test_parseqrun"),
    });

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let opt = RandomSearch::new(7);
    let cod = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(cod, func);
    let stop = Calls::new(50);
    let saver = CSVSaver::new("tmp_test_parseqrun", true, true, true, true, 1);

    let exp = experiment!(Threaded, RandomSearch | sp, obj, opt, stop, saver);
    exp.run();

    run_reader("tmp_test_parseqrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(cod, func);
    let saver = CSVSaver::new("tmp_test_parseqrun", true, true, true, true, 1);
    let mut exp = load!(Threaded, RandomSearch, Calls | sp, obj, saver);

    assert_eq!(exp.stop.0, 50, "Number of calls is wrong");
    assert_eq!(exp.optimizer.0.iteration, 8, "Number of iteration is wrong");
    assert_eq!(exp.optimizer.0.batch, 7, "Batch size is wrong");

    exp.stop.1 = 100;
    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(cod, func);
    let saver = CSVSaver::new("tmp_test_parseqrun", true, true, true, true, 1);
    let exp = load!(Threaded, RandomSearch, Calls | sp, obj, saver);
    run_reader("tmp_test_parseqrun", 100);
    assert_eq!(exp.stop.0, 100, "Number of calls is wrong");
    assert_eq!(
        exp.optimizer.0.iteration, 15,
        "Number of iteration is wrong"
    );
    assert_eq!(exp.optimizer.0.batch, 7, "Batch size is wrong");

    drop(Cleaner {
        path: String::from("tmp_test_parseqrun"),
    });
}
