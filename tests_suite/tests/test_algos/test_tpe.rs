use tantale::algos::bayesian::bandwidth::Optuna;
use tantale::algos::{Hyperopt, LinearSplit, Scott, SqrtSplit, Tpe, UniformWeighter, Univariate, tpe};
use tantale::core::Experiment;
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack,
    experiment::Runable,
    stop::Evaluated,
};

use crate::cleaner::Cleaner;
use crate::init_func::{sp_evaluator, sp_evaluator_real};
use crate::run_checker::{run_reader, run_reader_eps};

#[test]
fn test_tpe_seq_run() {
    let _clean = Cleaner::new("tmp_test_tpe_seqrun");

    let pb = sp_evaluator::get_problem();

    let opt = Tpe::new(
        5,
        10,
        (Univariate, Optuna::new(false)),
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_tpe_seqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader("tmp_test_tpe_seqrun", 50);

    let pb = sp_evaluator::get_problem();


    let config = FolderConfig::new("tmp_test_tpe_seqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, tpe!(Univariate, Optuna, UniformWeighter, LinearSplit), Evaluated, _, _>::load(pb, rec, check);

    let expstop: &mut Evaluated = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let pb = sp_evaluator::get_problem();


    let config = FolderConfig::new("tmp_test_tpe_seqrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let _exp = Experiment::<_, tpe!(Univariate, Optuna, UniformWeighter, LinearSplit), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_tpe_seqrun", 100);
}



#[test]
fn test_tpe_seq_run_real() {
    let _clean = Cleaner::new("tmp_test_tpe_seqrun_real");

    let pb = sp_evaluator_real::get_problem();
    let opt = Tpe::new(
        5,
        10,
        (Univariate, Optuna::new(false)),
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_tpe_seqrun_real");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader("tmp_test_tpe_seqrun_real", 50);

    let pb = sp_evaluator_real::get_problem();

    let config = FolderConfig::new("tmp_test_tpe_seqrun_real");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, tpe!(Univariate, Optuna, UniformWeighter, LinearSplit), Evaluated, _, _>::load(pb, rec, check);

    let expstop: &mut Evaluated = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let pb = sp_evaluator_real::get_problem();

    let config = FolderConfig::new("tmp_test_tpe_seqrun_real");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let _exp = Experiment::<_, tpe!(Univariate, Optuna, UniformWeighter, LinearSplit), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_tpe_seqrun_real", 100);
}


#[test]
fn test_tpe_seq_run_sqrt_splitter() {
    let _clean = Cleaner::new("tmp_test_tpe_seqrun_sqrt");

    let pb = sp_evaluator::get_problem();

    let opt = Tpe::new(
        5,
        10,
        (Univariate, Optuna::new(false)),
        UniformWeighter::default(),
        SqrtSplit::new(0.25).unwrap(),
    );
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_tpe_seqrun_sqrt");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader("tmp_test_tpe_seqrun_sqrt", 50);

    let pb = sp_evaluator::get_problem();


    let config = FolderConfig::new("tmp_test_tpe_seqrun_sqrt");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, tpe!(Univariate, Optuna, UniformWeighter, SqrtSplit), Evaluated, _, _>::load(pb, rec, check);

    let expstop: &mut Evaluated = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let pb = sp_evaluator::get_problem();


    let config = FolderConfig::new("tmp_test_tpe_seqrun_sqrt");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let _exp = Experiment::<_, tpe!(Univariate, Optuna, UniformWeighter, SqrtSplit), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_tpe_seqrun_sqrt", 100);
}

#[test]
fn test_tpe_seq_run_scott() {
    let _clean = Cleaner::new("tmp_test_tpe_seqrun_scott");

    let pb = sp_evaluator::get_problem();

    let opt = Tpe::new(
        5,
        10,
        (Univariate, Scott::new(false)),
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_tpe_seqrun_scott");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader("tmp_test_tpe_seqrun_scott", 50);

    let pb = sp_evaluator::get_problem();


    let config = FolderConfig::new("tmp_test_tpe_seqrun_scott");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, tpe!(Univariate, Scott, UniformWeighter, LinearSplit), Evaluated, _, _>::load(pb, rec, check);

    let expstop: &mut Evaluated = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let pb = sp_evaluator::get_problem();


    let config = FolderConfig::new("tmp_test_tpe_seqrun_scott");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let _exp = Experiment::<_, tpe!(Univariate, Scott, UniformWeighter, LinearSplit), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_tpe_seqrun_scott", 100);
}

#[test]
fn test_tpe_seq_run_hyperopt() {
    let _clean = Cleaner::new("tmp_test_tpe_seqrun_hyperopt");

    let pb = sp_evaluator::get_problem();

    let opt = Tpe::new(
        5,
        10,
        (Univariate, Hyperopt::new(false, false)),
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_tpe_seqrun_hyperopt");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader("tmp_test_tpe_seqrun_hyperopt", 50);

    let pb = sp_evaluator::get_problem();


    let config = FolderConfig::new("tmp_test_tpe_seqrun_hyperopt");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, tpe!(Univariate, Hyperopt, UniformWeighter, LinearSplit), Evaluated, _, _>::load(pb, rec, check);

    let expstop: &mut Evaluated = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let pb = sp_evaluator::get_problem();


    let config = FolderConfig::new("tmp_test_tpe_seqrun_hyperopt");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let _exp = Experiment::<_, tpe!(Univariate, Hyperopt, UniformWeighter, LinearSplit), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_tpe_seqrun_hyperopt", 100);
}

#[test]
fn test_tpe_seq_run_hyperopt_consider_endpoints() {
    let _clean = Cleaner::new("tmp_test_tpe_seqrun_hyperopt_endpoint");

    let pb = sp_evaluator::get_problem();

    let opt = Tpe::new(
        5,
        10,
        (Univariate, Hyperopt::new(true, false)),
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_tpe_seqrun_hyperopt_endpoint");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    run_reader("tmp_test_tpe_seqrun_hyperopt_endpoint", 50);

    let pb = sp_evaluator::get_problem();


    let config = FolderConfig::new("tmp_test_tpe_seqrun_hyperopt_endpoint");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, tpe!(Univariate, Hyperopt, UniformWeighter, LinearSplit), Evaluated, _, _>::load(pb, rec, check);

    let expstop: &mut Evaluated = exp.mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let pb = sp_evaluator::get_problem();


    let config = FolderConfig::new("tmp_test_tpe_seqrun_hyperopt_endpoint");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let _exp = Experiment::<_, tpe!(Univariate, Hyperopt, UniformWeighter, LinearSplit), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_tpe_seqrun_hyperopt_endpoint", 100);
}

// #[test]
// fn test_tpe_seqthr_run() {
//     let _clean = Cleaner::new("tmp_test_tpe_seqthrrun");

//     let pb = sp_evaluator::get_problem();

//     let opt = Tpe::new(
//         5,
//         10,
//         (Univariate, Optuna::new(false)),
//         UniformWeighter::default(),
//         LinearSplit::new(0.25).unwrap(),
//     );
//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_tpe_seqthrrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     run_reader_eps("tmp_test_tpe_seqthrrun", 50, num_cpus::get() * 4);

//     let pb = sp_evaluator::get_problem();


//     let config = FolderConfig::new("tmp_test_tpe_seqthrrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         tpe!(Univariate, Optuna, UniformWeighter, LinearSplit),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );

//     let expstop: &mut Evaluated = exp.mut_stop();
//     assert!(
//         expstop.calls() >= 50 && expstop.calls() <= 50 + num_cpus::get(),
//         "Number of calls is wrong"
//     );
//     expstop.add(50);

//     exp.run();

//     let pb = sp_evaluator::get_problem();


//     let config = FolderConfig::new("tmp_test_tpe_seqthrrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let _exp = load!(
//         threaded,
//         tpe!(Univariate, Optuna, UniformWeighter, LinearSplit),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );
//     run_reader_eps("tmp_test_tpe_seqthrrun", 100, num_cpus::get() * 4);
// }


// #[test]
// fn test_tpe_seqthr_run_real() {
//     let _clean = Cleaner::new("tmp_test_tpe_seqthrrun_real");

//     let sp = sp_evaluator_real::get_searchspace();
//     let func = sp_evaluator_real::example;
//     let opt = Tpe::new(
//         5,
//         10,
//         (Univariate, Optuna::new(false)),
//         UniformWeighter::default(),
//         LinearSplit::new(0.25).unwrap(),
//     );
//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_tpe_seqthrrun_real");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     run_reader_eps("tmp_test_tpe_seqthrrun_real", 50, num_cpus::get() * 4);

//     let sp = sp_evaluator_real::get_searchspace();
//     let func = sp_evaluator_real::example;

//     let config = FolderConfig::new("tmp_test_tpe_seqthrrun_real");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         tpe!(Univariate, Optuna, UniformWeighter, LinearSplit),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );

//     let expstop: &mut Evaluated = exp.mut_stop();
//     assert!(
//         expstop.calls() >= 50 && expstop.calls() <= 50 + num_cpus::get(),
//         "Number of calls is wrong"
//     );
//     expstop.add(50);

//     exp.run();

//     let sp = sp_evaluator_real::get_searchspace();
//     let func = sp_evaluator_real::example;

//     let config = FolderConfig::new("tmp_test_tpe_seqthrrun_real");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let _exp = load!(
//         threaded,
//         tpe!(Univariate, Optuna, UniformWeighter, LinearSplit),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );
//     run_reader_eps("tmp_test_tpe_seqthrrun_real", 100, num_cpus::get() * 4);
// }