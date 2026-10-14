use super::init_sp::sp_mixed_to_nodomain::get_searchspace;

use csv::Reader;
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc};
use tantale::core::{
    Codomain, Computed, EmptyInfo, FidelitySol, FolderConfig, HasFidelity, HasId, HasStep, HasX,
    Mixed, MixedTypeDom, MultiStep, NoDomain, Problem, Recordable, Recorder, Schema, Searchspace,
    Sp, Step, StepSId, HasVariables, recorder::RecorderTargets,
    solution::Fidelity, utils::record::{CompParts, SolParts},
};
use tantale::core::recorder::CSVRecorder;
use tantale::core::FidPb;
use tantale::macros::{Outcome, Recordable, Schema};

use crate::init_func::FnState;

mod infos {
    use super::*;

    #[derive(Outcome, Recordable, Schema, Debug, Serialize, Deserialize)]
    pub struct FidOutExample {
        pub fid2: usize,
        pub con3: i64,
        #[constraint]
        pub con4: f64,
        #[constraint]
        pub con5: f64,
        #[maximize]
        pub mul6: f64,
        #[maximize]
        pub mul7: f64,
        #[maximize]
        pub mul8: f64,
        #[maximize]
        pub mul9: f64,
        pub tvec: Vec<f64>,
        #[step]
        pub state: Step,
    }

    pub fn get_out(fid: usize, a: i64) -> FidOutExample {
        FidOutExample {
            fid2: fid,
            con3: a,
            con4: 4.4,
            con5: 5.5,
            mul6: 6.6,
            mul7: 7.7,
            mul8: 8.8,
            mul9: 9.9,
            tvec: Vec::from([1.1, 2.2, 3.3]),
            state: Step::Partially(1),
        }
    }
}

fn read_csv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let mut reader = Reader::from_path(path).unwrap();
    let headers = reader
        .headers()
        .unwrap()
        .iter()
        .map(str::to_owned)
        .collect();
    let rows = reader
        .records()
        .map(|record| record.unwrap().iter().map(str::to_owned).collect())
        .collect();
    (headers, rows)
}

#[test]
fn csv_recorder_writes_fidelity_schema_and_records() {
    let _cleaner = Cleaner;
    type Sol = FidelitySol<StepSId, Mixed, EmptyInfo>;
    type Func = MultiStep<Sol, infos::FidOutExample, FnState>;
    type Pb = FidPb<EmptyInfo, Sp<Mixed, NoDomain>, infos::FidOutExample, FnState>;

    let searchspace: Sp<Mixed, NoDomain> = get_searchspace();
    let objective = MultiStep::new(
        |_x: Arc<[MixedTypeDom]>, _fidelity: Fidelity, _state: Option<FnState>| {
            (infos::get_out(0, 0), FnState { state: 0 })
        },
    );
    let problem: Pb = Pb::new(searchspace, objective);
    let config = FolderConfig::new("tmp_test_fid");
    let mut recorder = CSVRecorder::new(config).unwrap();
    <CSVRecorder as Recorder<_, ()>>::init(&mut recorder, &problem, &()).unwrap();

    let mut rng = rand::rng();
    let mut obj: Sol = <Sp<Mixed, NoDomain> as Searchspace<Func>>::sample_obj(problem.get_searchspace(),&mut rng);
    
    obj.set_fidelity(0.75);
    obj.partially(1);
    let opt = obj.clone();
    let id = obj.id();
    let int_value = match obj.ref_x()[0] {
        MixedTypeDom::Int(value) => value,
        _ => panic!("The first searchspace variable should be an integer."),
    };
    let output = infos::get_out(id.id, int_value);
    let codomain = problem.get_codomain();
    let computed: Computed<FidelitySol<StepSId, Mixed, EmptyInfo>, infos::FidOutExample> = Computed::new(obj.clone(), Arc::new(codomain.get_elem(&output)));
    let vars = problem.get_searchspace().get_vars();
    let contexts = (codomain, &(vars, codomain));

    <CSVRecorder as Recorder<FidPb<EmptyInfo, Sp<Mixed, NoDomain>, infos::FidOutExample, FnState>, ()>>::record(&recorder, &RecorderTargets::XObjective, &obj, (&(), &vars), SolParts::X).unwrap();
    <CSVRecorder as Recorder<FidPb<EmptyInfo, Sp<Mixed, NoDomain>, infos::FidOutExample, FnState>, ()>>::record(&recorder, &RecorderTargets::XOptimizer, &opt, (&(), &vars), SolParts::X).unwrap();
    <CSVRecorder as Recorder<FidPb<EmptyInfo, Sp<Mixed, NoDomain>, infos::FidOutExample, FnState>, ()>>::record(&recorder, &RecorderTargets::Y, &computed, contexts, CompParts::Y).unwrap();
    <CSVRecorder as Recorder<FidPb<EmptyInfo, Sp<Mixed, NoDomain>, infos::FidOutExample, FnState>, ()>>::record(&recorder, &RecorderTargets::Outcome, &output, (&(), &()), ()).unwrap();
    <CSVRecorder as Recorder<FidPb<EmptyInfo, Sp<Mixed, NoDomain>, infos::FidOutExample, FnState>, ()>>::record(&recorder, &RecorderTargets::Metadata,&obj,(&(), &vars),SolParts::Info).unwrap();

    let recorder_dir = Path::new("tmp_test_fid").join("recorder");
    let (obj_header, obj_rows) = read_csv(&recorder_dir.join("obj.csv"));
    assert_eq!(obj_header, Sol::columns(&vars, SolParts::X));
    assert_eq!(obj_rows, vec![obj.record(&(), SolParts::X)]);

    let (opt_header, opt_rows) = read_csv(&recorder_dir.join("opt.csv"));
    assert_eq!(opt_header, Sol::columns(&vars, SolParts::X));
    assert_eq!(opt_rows, vec![opt.record(&(), SolParts::X)]);

    let (cod_header, cod_rows) = read_csv(&recorder_dir.join("cod.csv"));
    assert_eq!(
        cod_header,
        Computed::<Sol, infos::FidOutExample>::columns(&(vars, codomain), CompParts::Y)
    );
    assert_eq!(cod_rows, vec![computed.record(codomain, CompParts::Y)]);

    let (out_header, out_rows) = read_csv(&recorder_dir.join("out.csv"));
    assert_eq!(out_header, infos::FidOutExample::columns(&(), ()));
    assert_eq!(out_rows, vec![output.record(&(), ())]);

    let (info_header, info_rows) = read_csv(&recorder_dir.join("info.csv"));
    assert_eq!(info_header, Sol::columns(&vars, SolParts::Info));
    assert_eq!(info_rows, vec![obj.record(&(), SolParts::Info)]);
    assert_eq!(obj.fidelity().0, 0.75);
    assert_eq!(obj.step(), Step::Partially(1));
}

struct Cleaner;

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all("tmp_test_fid");
    }
}