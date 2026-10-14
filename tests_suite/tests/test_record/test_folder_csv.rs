use super::init_sp::sp_mixed_to_nodomain::get_searchspace;

use csv::Reader;
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc};
use tantale::core::{
    BaseSol, Codomain, Computed, EmptyInfo, FolderConfig, HasId, HasVariables, HasX, 
    Mixed, MixedTypeDom, NoDomain, OneStep, Problem, Recordable, Recorder, SId, Schema, Searchspace, SimplePb, Sp, 
    recorder::RecorderTargets, utils::record::{CompParts, SolParts},
};
use tantale::core::recorder::CSVRecorder;
use tantale::macros::{Outcome, Recordable, Schema};

mod infos {
    use super::*;

    #[derive(Outcome, Recordable, Schema, Debug, Serialize, Deserialize)]
    pub struct OutExample {
        pub fid2: usize,
        pub con3: i64,
        pub con4: f64,
        pub con5: f64,
        #[maximize]
        pub mul6: f64,
        pub mul7: f64,
        pub mul8: f64,
        pub mul9: f64,
        pub tvec: Vec<f64>,
    }

    pub fn get_out(fid: usize, a: i64) -> OutExample {
        OutExample {
            fid2: fid,
            con3: a,
            con4: 4.4,
            con5: 5.5,
            mul6: 6.6,
            mul7: 7.7,
            mul8: 8.8,
            mul9: 9.9,
            tvec: Vec::from([1.1, 2.2, 3.3]),
        }
    }

    pub fn evaluate(_x: Arc<[MixedTypeDom]>) -> OutExample {
        get_out(0, 0)
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
fn csv_recorder_writes_schema_and_records() {
    let searchspace: Sp<Mixed, NoDomain> = get_searchspace();
    let objective: OneStep<BaseSol<SId, Mixed, EmptyInfo>, infos::OutExample> =
        OneStep::new(infos::evaluate);
    let problem: SimplePb<EmptyInfo, Sp<Mixed, NoDomain>, infos::OutExample> =
        SimplePb::new(searchspace, objective);
    let config = FolderConfig::new("tmp_test");
    let mut recorder = CSVRecorder::new(config).unwrap();
    <CSVRecorder as Recorder<_, ()>>::init(&mut recorder, &problem, &()).unwrap();

    let mut rng = rand::rng();
    let solution: BaseSol<SId, Mixed, EmptyInfo> = <Sp<Mixed, NoDomain> as Searchspace<OneStep<BaseSol<SId, Mixed, EmptyInfo>, infos::OutExample>>>::sample_obj(problem.get_searchspace(),&mut rng);
    let id = solution.id();
    let int_value = match solution.ref_x()[0] {
        MixedTypeDom::Int(value) => value,
        _ => panic!("The first searchspace variable should be an integer."),
    };
    let output = infos::get_out(id.id, int_value);
    let codomain = problem.get_codomain();
    let computed: Computed<BaseSol<SId, Mixed, EmptyInfo>, infos::OutExample> = Computed::new(solution.clone(), Arc::new(codomain.get_elem(&output)));
    let vars = problem.get_searchspace().get_vars();
    let contexts = (codomain, &(vars, codomain));

    <CSVRecorder as Recorder<SimplePb<EmptyInfo, Sp<Mixed, NoDomain>, infos::OutExample>, ()>>::record(&recorder, &RecorderTargets::XObjective,&solution,(&(), &vars),SolParts::X).unwrap();
    <CSVRecorder as Recorder<SimplePb<EmptyInfo, Sp<Mixed, NoDomain>, infos::OutExample>, ()>>::record(&recorder,&RecorderTargets::XOptimizer,&solution,(&(), &vars),SolParts::X,).unwrap();
    <CSVRecorder as Recorder<SimplePb<EmptyInfo, Sp<Mixed, NoDomain>, infos::OutExample>, ()>>::record(&recorder,&RecorderTargets::Y, &computed, contexts, CompParts::Y).unwrap();
    <CSVRecorder as Recorder<SimplePb<EmptyInfo, Sp<Mixed, NoDomain>, infos::OutExample>, ()>>::record(&recorder,&RecorderTargets::Outcome, &output, (&(), &()), ()).unwrap();
    <CSVRecorder as Recorder<SimplePb<EmptyInfo, Sp<Mixed, NoDomain>, infos::OutExample>, ()>>::record(&recorder,&RecorderTargets::Metadata,&solution,(&(), &vars),SolParts::Info).unwrap();

    let recorder_dir = Path::new("tmp_test").join("recorder");
    let obj_file = recorder_dir.join("obj.csv");
    let (obj_header, obj_rows) = read_csv(&obj_file);
    assert_eq!(
        obj_header,
        BaseSol::<SId, Mixed, EmptyInfo>::columns(&vars, SolParts::X)
    );
    assert_eq!(obj_rows, vec![solution.record(&(), SolParts::X)]);

    let opt_file = recorder_dir.join("opt.csv");
    let (opt_header, opt_rows) = read_csv(&opt_file);
    assert_eq!(
        opt_header,
        BaseSol::<SId, Mixed, EmptyInfo>::columns(&vars, SolParts::X)
    );
    assert_eq!(opt_rows, vec![solution.record(&(), SolParts::X)]);

    let cod_file = recorder_dir.join("cod.csv");
    let (cod_header, cod_rows) = read_csv(&cod_file);
    assert_eq!(
        cod_header,
        Computed::<BaseSol<SId, Mixed, EmptyInfo>, infos::OutExample>::columns(
            &(vars, codomain),
            CompParts::Y,
        )
    );
    assert_eq!(cod_rows, vec![computed.record(codomain, CompParts::Y)]);

    let out_file = recorder_dir.join("out.csv");
    let (out_header, out_rows) = read_csv(&out_file);
    assert_eq!(out_header, infos::OutExample::columns(&(), ()));
    assert_eq!(out_rows, vec![output.record(&(), ())]);

    let info_file = recorder_dir.join("info.csv");
    let (info_header, info_rows) = read_csv(&info_file);
    assert_eq!(
        info_header,
        BaseSol::<SId, Mixed, EmptyInfo>::columns(&vars, SolParts::Info)
    );
    assert_eq!(info_rows, vec![solution.record(&(), SolParts::Info)]);
}