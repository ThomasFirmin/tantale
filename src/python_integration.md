# Python objective integration

The optional `py` feature lets a Rust experiment call a Python objective. The
Rust side declares the search space with [`pyhpo!`](crate::macros::pyhpo), then [`init_python!`](crate::python::init_python) loads
the Python callable and registers it as a Tantale objective. Python-backed
outcomes implement `header()` and `record()` for recording. These are the
method names used by the current Python bridge.

Add the feature to the dependency:

```toml
[dependencies]
tantale = { version = "0.3.2", features = ["py"] }
serde = { version = "1", features = ["derive"] }
```

The Python files are resolved relative to `CARGO_MANIFEST_DIR`. The name
provided for a Python module must match the importable module name for that
file. Install Python and the packages imported by the objective in the
environment used by the Rust process. Call `init_python!` once before
spawning worker threads or processes.

## Python side

For this example, save the following as `src/objective.py`:

```python
from pytantale import indices

class Evaluation:
    def __init__(self, score: float, extra: float):
        self.score = score
        self.extra = extra

    @staticmethod
    def header() -> list[str]:
        return ["score", "extra"]

    def record(self) -> list[str]:
        return [str(self.score), str(self.extra)]


def objective(x: list) -> Evaluation:
    value = float(x[indices.A])
    return Evaluation(score=1.0 - abs(value - 0.25), extra=value)
```

The class attributes named in `objectives` must be numeric, because Tantale
reads them as optimization metrics. `header()` supplies output column names;
`record()` supplies the corresponding string values. The one-step objective
returns an instance of the registered outcome class.

## Rust setup

The following shows the current macro form. The runnable Python integration
tests in `tests_python/` and examples under `examples/examples/` contain
complete project wiring.

```rust,ignore
mod searchspace {
    use tantale::core::{Real, Uniform};
    use tantale::macros::pyhpo;

    pyhpo! {
        a | Real(0.0, 1.0, Uniform) | Real(0.0, 1.0, Uniform);
    }
}

use tantale::python::init_python;
use tantale::algos::RandomSearch;
use tantale::core::{CSVRecorder, Experiment, FolderConfig, MessagePack, Pb, Runable, stop::Evaluated};

fn main(){
    let searchspace = searchspace::get_searchspace();
    let objective = init_python!(
        OneStep, searchspace,
        "src/objective.py", "objective", "objective",
        "src/objective.py", "objective", "Evaluation",
        objectives: [maximize "score"]
    );
    let problem = Pb::new(searchspace, objective);
    let sampler = RandomSearch::new();
    let stop = Evaluated::new(20);
    let config = FolderConfig::new("tmp_python_objective_run");
    let recorder = CSVRecorder::new(config.clone());
    let checkpointer = MessagePack::new(config);
    let _accumulator = Experiment::new(problem, sampler, stop, recorder, checkpointer).run();
    # let _ = std::fs::remove_dir_all("tmp_python_objective_run");
}
```

## Multi-fidelity Python objective

Unlike the one-step function, a multi-step function accepts a floating-point
fidelity and an optional state and returns `(outcome, state)`. Here fidelity
is a total epoch budget, not the number of additional epochs. For a real
trainer the state must also contain model weights and optimizer state.

Save this example as `src/training.py`:

```python
from pathlib import Path
import pytantale as tnt
from pytantale import indices


class State:
    def __init__(self, epochs: int = 0):
        self.epochs = epochs

    def save(self, path: str) -> None:
        Path(path, "epochs.txt").write_text(str(self.epochs))

    @staticmethod
    def load(path: str) -> "State":
        return State(int(Path(path, "epochs.txt").read_text()))


class Evaluation:
    def __init__(self, score: float, cost: float, step: tnt.PyStep):
        self.score = score
        self.cost = cost
        self.step = step

    @staticmethod
    def header() -> list[str]:
        return ["score", "cost", "step"]

    def record(self) -> list[str]:
        return [str(self.score), str(self.cost), str(self.step)]


def train(x: list, fidelity: float, state: State | None) -> tuple[Evaluation, State]:
    state = state if state is not None else State()
    target_epoch = int(fidelity)
    # A real trainer would resume from state.epochs to target_epoch.
    state.epochs = target_epoch
    value = float(x[indices.A])
    step = tnt.PyStep.evaluated() if fidelity >= 4.0 else tnt.PyStep.partially(target_epoch)
    return Evaluation(value, value * value, step), state
```

The [`State`](crate::core::State) class is defined at module scope so it can be pickled. The bridge
calls its `save(path)` and `load(path)` methods for persistent state. Here is
the corresponding Rust setup; use the `searchspace` module defined above:

```rust,ignore
use tantale::algos::{MoAsha, RandomSearch, mo::NSGA2Selector};
use tantale::core::{CSVRecorder, Calls, FolderConfig, MessagePack, ParExperiment, Pb, ThreadCommunicator};
use tantale::python::init_python;

fn main() {
    let sp = searchspace::get_searchspace();
    let objective = init_python!(
        MultiStep, searchspace,
        "src/training.py", "training", "train",
        "src/training.py", "training", "Evaluation",
        objectives: [maximize "score", minimize "cost"],
        step: "step",
    );
    let problem = Pb::new(sp, objective);
    let sampler = MoAsha::new(RandomSearch::new(), NSGA2Selector, 1.0, 4.0, 2.0)?;
    let config = FolderConfig::new("python_multifidelity_run");
    let recorder = CSVRecorder::new(config.clone());
    let checkpointer = MessagePack::new(config);
    let _accumulator = Experiment::new(problem, sampler, stop, recorder, checkpointer).run();
}
```

`step: "step"` selects the Python **attribute** read by the macro; it must
contain a [`PyStep`](crate::python::PyStep). Partial steps must be positive, and the final budget must
return `PyStep.evaluated()`. For a detailed explanation of promotion and
fidelity, see the [MO-ASHA tutorial](mo_example.md).

The examples above require the `py` feature and an embedded Python runtime;
they are not run by the facade RustDoc tests. Both examples were compiled and
run in an isolated project; the multi-fidelity run recorded a promotion.
Some older workspace Python examples still use `csv_header` and `csv_write`;
the current bridge calls `header` and `record`.