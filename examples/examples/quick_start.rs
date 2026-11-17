mod searchspace {
    use serde::{Deserialize, Serialize};
    use tantale::core::{Real, Uniform};
    use tantale::macros::{Outcome, Recordable, Schema, objective};

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct Evaluation {
        #[maximize]
        pub score: f64,
        pub extra: f64,
    }

    objective!(
        pub fn tune() -> Evaluation {
            let x = [! x | Real(0.0, 1.0, Uniform) | !];
            let y = [! y | Real(0.0, 1.0, Uniform) | !];
            Evaluation {
                score: 1.0 - (x - 0.25).abs() - (y - 0.75).abs(),
                extra: x + y,
            }
        }
    );
}
use searchspace::get_problem;
use tantale::algos::RandomSearch;
use tantale::core::{
    CSVRecorder, Experiment, FolderConfig, HasX, HasY, MessagePack, SolutionShape,
    experiment::Runable, stop::Evaluated,
};

fn main() {
    let problem = get_problem();
    let sampler = RandomSearch::new();
    let stop = Evaluated::new(50);

    let config = FolderConfig::new("tmp_quick_start");
    let recorder = CSVRecorder::new(config.clone());
    let checkpointer = MessagePack::new(config);

    let experiment = Experiment::new(
        problem,
        sampler,
        stop,
        recorder,
        checkpointer,
    );
    let accumulator = experiment.run();
    if let Some(best) = accumulator.get().map(|acc| acc.get_sobj()) {
        println!("Best candidate: {:?}, score: {}", best.ref_x(), best.y().value);
    }
    let _ = std::fs::remove_dir_all("tmp_quick_start");
}