use tantale::algos::{Sha, sha};
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack, SaverConfig,
    experiment::{Runable, mono, threaded},
    load,
    stop::Evaluated,
};

use crate::cleaner::Cleaner;
use crate::init_func::{FidOutEvaluator, sp_evaluator_sh};
use crate::run_checker::run_reader;

#[test]
fn test_fid_batch_run() {
    let _clean = Cleaner::new("tmp_test_sh_run");

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let opt = Sha::new(10, 1., 5., 1.61); // log(max/min)
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_sh_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_sh_run", 1000);

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_sh_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, Sha, Evaluated, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.iteration, 200,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.0.batch, 10, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_sh_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, Sha, Evaluated, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_sh_run", 2000);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.iteration, 400,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.0.batch, 10, "Batch size is wrong");
}

#[test]
fn test_fid_batch_parrun() {
    let _clean = Cleaner::new("tmp_test_sh_parrun");

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let opt = Sha::new(10, 1., 5., 1.61);
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_sh_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_sh_parrun", 1000);

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_sh_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(threaded, Sha, Evaluated, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.get_mut_optimizer();
    assert_eq!(
        expoptimizer.0.iteration, 200,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.0.batch, 10, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_sh_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(threaded, Sha, Evaluated, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_sh_parrun", 2000);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.iteration, 400,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.0.batch, 10, "Batch size is wrong");
}
