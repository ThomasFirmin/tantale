#[path = "./cleaner.rs"]
pub mod cleaner;
pub use cleaner::Cleaner;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = get_problem();
    let sampler = RandomSearch::new();
    let stop = Evaluated::new(50);

    let run_dir = format!("random-search-run-{}", std::process::id());
    let _cleaner = Cleaner::new(&run_dir);
    let config = FolderConfig::new(&run_dir);
    let recorder = CSVRecorder::new(config.clone())
        .ok_or_else(|| std::io::Error::other("could not create CSV recorder"))?;
    let checkpointer = MessagePack::new(config)
        .ok_or_else(|| std::io::Error::other("could not create MessagePack checkpointer"))?;

    let experiment = Experiment::new(
        problem,
        sampler,
        stop,
        Some(recorder),
        Some(checkpointer),
    );
    let accumulator = experiment.run();
    if let Some(best) = accumulator.get().map(|acc| acc.get_sobj()) {
        println!("Best candidate: {:?}, score: {}", best.ref_x(), best.y().value);
    }
    Ok(())
}