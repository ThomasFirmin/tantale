//! # Tantale algorithms
//!
//! `tantale_algos` provides concrete samplers and pruning strategies for
//! [`tantale_core`](https://docs.rs/tantale_core). It is also re-exported by
//! the `tantale` facade crate as `tantale::algos`.
//!
//! ## Installation and features
//!
//! Add the algorithms crate directly when building a Rust integration:
//!
//! ```toml
//! [dependencies]
//! tantale_algos = "0.3.2"
//! tantale_core = "0.3.2"
//! ```
//!
//! The facade crate re-exports both crates and includes macros for declaring
//! objectives:
//!
//! ```toml
//! [dependencies]
//! tantale = "0.3.2"
//! ```
//!
//! The crate uses Rust 2024 and requires Rust 1.91.1 or later. Feature flags:
//!
//! | Feature | Effect |
//! | --- | --- |
//! | `bayes` | Enables the TPE implementation and its statistical components. |
//! | `mpi` | Enables the MPI support forwarded to `tantale_core`. |
//!
//! The default feature set does not include `bayes` or `spikes`.
//!
//! ## How the algorithms fit into Tantale
//!
//! The algorithms in this crate implement the core
//! [`Sampler`](tantale_core::Sampler) interface.
//! A sampler proposes an unevaluated candidate and receives computed
//! candidates through its update method. The core
//! [`Experiment`](tantale_core::Experiment) runs a problem with a sampler and
//! a [`Stop`](tantale_core::Stop) criterion; the experiment owns
//! objective evaluation, result accumulation, recording, and checkpointing.
//!
//! [Pruners](tantale_core::Pruner) are samplers dedicated to multi-fidelity optimization.
//! They promote promising partial evaluations to higher fidelity and discard
//! candidates according to their ranking. A multi-fidelity problem must
//! expose fidelity and step information in its solution and outcome types.
//! Use [`NoPruner`] when a function has multiple evaluation steps but the
//! optimizer should simply continue the same candidate until it is complete.
//!
//! ## Algorithm selection
//!
//! | Algorithm | Use |
//! | --- | --- |
//! | [`RandomSearch`] | Samples candidates on demand from the search space. |
//! | [`BatchRandomSearch`] | Samples a configurable batch of random candidates. |
//! | [`GridSearch`] | Enumerates a Cartesian product of finite grid domains. |
//! | [`Sha`] | Runs synchronous Successive Halving over fixed-size batches. |
//! | [`Asha`] | The asynchronous version of [`Sha`]. |
//! | [`MoAsha`] | Multi-objective ASHA using [`CandidateSelector`](mo::CandidateSelector). |
//! | [`Hyperband`] | Wraps [`Sha`] in a sequence of resource-allocation brackets.* |
//! | [`Tpe`] | Proposes candidates from density models, when `bayes` is enabled. |
//! | [`NoPruner`] | Resumes a partial candidate without pruning. |
//! | [`Premade`] | Evaluates a supplied ordered list of candidates. |
//!
//! *`Sha::into_hyperband` creates the `Hyperband` wrapper. `Asha::into_hyperband`
//! and `MoAsha::into_hyperband` create their bracketed asynchronous variants;
//! these are separate types from [`Hyperband`].
//!
//! ## Random-search example
//!
//! This complete example defines a one-variable objective and runs a
//! `RandomSearch` experiment. The example is compiled and executed as a
//! RustDoc test. The facade crate supplies the objective and outcome macros;
//! the algorithm type is imported from `tantale_algos`.
//!
//! ```
//! use tantale::core::{
//!     Evaluated, CSVRecorder, Experiment, FolderConfig, MessagePack,
//!     experiment::Runable,
//! };
//! use tantale_algos::RandomSearch;
//!
//! mod model {
//!     use serde::{Deserialize, Serialize};
//!     use tantale::core::{Real, Uniform};
//!     use tantale::macros::{Outcome, Recordable, Schema};
//!     use tantale::objective;
//!
//!     #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
//!     pub struct Score {
//!         #[maximize]
//!         value: f64,
//!     }
//!
//!     objective!(
//!         pub fn quadratic() -> Score {
//!             let x = [! x | Real(-2.0, 2.0, Uniform) | !];
//!             Score {
//!                 value: -(x - 0.25).powi(2),
//!             }
//!         }
//!     );
//! }
//!
//! let problem = model::get_problem();
//! let output = std::env::temp_dir().join(format!(
//!     "tantale-algos-doc-{}",
//!     std::process::id()
//! ));
//! let config = FolderConfig::new(output.to_str().expect("path is valid UTF-8"));
//! let experiment = Experiment::new(
//!     problem,
//!     RandomSearch::new(),
//!     Evaluated::new(8),
//!     CSVRecorder::new(config.clone()),
//!     MessagePack::new(config),
//! );
//! let result = experiment.run();
//! assert!(result.get().is_some());
//! std::fs::remove_dir_all(output).expect("remove temporary experiment files");
//! ```
//!
//!
//! The `examples/` directory in the workspace contains complete programs for
//! random search, grid search, multi-fidelity optimization, Bayesian
//! optimization, multi-objective optimization, and threaded or MPI execution.
//!
//! ## Public modules
//!
//! - [`random_search`](mod@random_search): random and batched random samplers and their metadata.
//! - [`grid_search`](mod@grid_search): finite-grid enumeration.
//! - [`sha`](mod@sha), [`asha`](mod@asha), [`hyperband`](mod@hyperband), and [`moasha`](mod@moasha): multi-fidelity pruning
//!   algorithms and bracketed variants.
//! - [`nopruner`](mod@nopruner): multi-step evaluation without candidate pruning.
//! - [`premade`](mod@premade): samplers backed by an explicit list of candidate inputs.
//! - [`utils`](mod@utils): shared rung, budget, sampling, and multi-objective selection
//!   helpers.
//! - [`bayesian`](mod@bayesian): TPE components, available with `bayes`.

pub mod random_search;
pub use random_search::{BatchRandomSearch, RSInfo, RandomSearch};

pub mod sha;
pub use sha::Sha;

pub mod asha;
pub use asha::Asha;

pub mod hyperband;
pub use hyperband::Hyperband;

pub mod moasha;
pub use moasha::MoAsha;

pub mod grid_search;
pub use grid_search::GridSearch;

pub mod utils;
pub use utils::mo;

#[cfg(feature = "bayes")]
pub mod bayesian;
#[cfg(feature = "bayes")]
pub use bayesian::{
    bandwidth::{Bandwidth, BandwidthType, CategoricalBw, Hyperopt, Optuna, Scott},
    kernel::Univariate,
    prior::BetaPrior,
    splitter::{LinearSplit, SqrtSplit},
    tpe,
    tpe::Tpe,
    weighter::UniformWeighter,
};

#[cfg(feature = "spikes")]
pub mod spikes;

pub mod nopruner;
pub use nopruner::NoPruner;

pub mod premade;
pub use premade::Premade;
