use std::path::Path;

use csv::Reader;
use tantale::algos::Premade;
use tantale::algos::{MoAsha, mo::NSGA2Selector};
use tantale::core::{CSVRecorder, Evaluated, Experiment, FolderConfig, MessagePack, Pb, Runable};
use tantale::python::init_python;

use crate::cleaner::Cleaner;

fn assert_moasha_promotion(path: &str) {
    let info_path = Path::new(path).join("recorder/info.csv");
    let mut readear = Reader::from_path(info_path).unwrap();
    let headers = readear.headers().unwrap().clone();
    let name_idx = headers.iter().position(|header| header == "Name").unwrap();

    let mut sequence = Vec::new();
    for row in readear.records() {
        let row = row.unwrap();
        let name = &row[name_idx];
        let names = vec![
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
        ];
        assert!(
            names.contains(&name),
            "Candidate {name} is not one of the expected candidates"
        );
        sequence.push(name.to_string());
    }

    assert_eq!(
        sequence,
        vec![
            "A", "K", "A", "B", "B", "B", "L", "F", "E", "E", "D", "D", "A", "D", "H", "G", "C",
            "C", "I", "J", "F", "M", "N", "G", "O", "P", "P", "C",
        ],
        "The sequence of candidates is not as expected"
    );
}

#[test]
fn test_moasha_promotion() {
    pub mod sp_ms_nosamp {
        use tantale::core::{distribution::Uniform, domain::Real};
        use tantale::macros::pyhpo;

        pyhpo! {
            a | Real(0.0, 10.0, Uniform)  | ;
            b | Real(0.0, 100.0, Uniform) | ;
            c | Real(0.0, 100.0, Uniform) | ;
        }
    }

    let _clean = Cleaner::new("tmp_py_test_moasha_promotion");

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        MultiStep, sp_ms_nosamp,
        "/tests/test_pytantale_stepped_async_run_promotion/function_fid_async_run_promotion.py", "function_fid_async_run_promotion", "objective",
        "/tests/test_pytantale_stepped_async_run_promotion/function_fid_async_run_promotion.py", "function_fid_async_run_promotion", "MyOutcome",
        objectives: [maximize "obj1", maximize "obj2"],
        step: "step",
    );

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 1.61_f64.powi(i))
        .take_while(|&b| b <= 5.)
        .collect();
    if *budgets.last().unwrap() != 5. {
        let last = budgets.last_mut().unwrap();
        *last = 5.;
    }

    let pb = Pb::new(sp, obj);

    let solutions = vec![
        (vec![0.1, 10., 6.], "A".to_string()),
        (vec![0.11, 1., 2.], "K".to_string()),
        (vec![0.2, 8., 7.], "B".to_string()),
        (vec![0.12, 1., 2.], "L".to_string()),
        (vec![0.6, 3., 4.], "F".to_string()),
        (vec![0.5, 4., 5.], "E".to_string()),
        (vec![0.4, 5., 9.], "D".to_string()),
        (vec![0.8, 1., 2.], "H".to_string()),
        (vec![0.7, 2., 3.], "G".to_string()),
        (vec![0.3, 6., 8.], "C".to_string()),
        (vec![0.8, 1., 2.], "I".to_string()),
        (vec![0.9, 1., 2.], "J".to_string()),
        (vec![0.13, 1., 2.], "M".to_string()),
        (vec![0.14, 1., 2.], "N".to_string()),
        (vec![0.15, 1., 2.], "O".to_string()),
        (vec![0.15, 1., 2.], "P".to_string()),
        (vec![0.15, 1., 2.], "Q".to_string()),
    ];

    let sampler = Premade::new(solutions, false);
    let opt = MoAsha::new(sampler, NSGA2Selector, 1., 4., 2.).unwrap(); // log(max/min)

    let stop = Evaluated::new(4);
    let config = FolderConfig::new("tmp_py_test_moasha_promotion");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    assert_moasha_promotion("tmp_py_test_moasha_promotion");
}
