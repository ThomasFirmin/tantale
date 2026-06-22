//! This crate contains some algorithms implemented in Tantale,
//! using the core library.
//!
//! The following algorithms are implemented:
//! - [Grid Search](mod@grid_search) :
//!   * [`GridSearch`] : the usual [Grid Search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search) algorithm.
//! - [Random Search](mod@random_search) :
//!   * [`RandomSearch`] : the usual sequential [Random Search](https://en.wikipedia.org/wiki/Random_search) algorithm. Generating points on demand.
//!   * [`BatchRandomSearch`] : a batch version of random search. Generating points in batches.
//! - [Successive Halving](mod@sha) :
//!   * [`Sha`] : the original [Successive Halving](https://arxiv.org/pdf/1502.07943) algorithm for multi-fidelity hyperparameter optimization.
//! - [Asynchronous Successive Halving](mod@asha) :
//!   * [`Asha`] : the asynchronous version of [Successive Halving](https://arxiv.org/pdf/1810.05934) algorithm for multi-fidelity hyperparameter optimization.
//! - [Hyperband](mod@hyperband) :
//!   * [`Hyperband`] : the original [Hyperband](https://arxiv.org/pdf/1603.06560) algorithm for multi-fidelity hyperparameter optimization.
//! - [MO-ASHA](mod@moasha) :
//!   * [`MoAsha`] : the multi-objective version of ASHA, based on the [MO-ASHA](https://arxiv.org/pdf/2106.12639) algorithm for multi-fidelity
//!     and multi-objective hyperparameter optimization.
//! - [Tree-structured Parzen Estimator](mod@bayesian::tpe) :
//!   * [`tpe::Tpe`] : the original [Tree-structured Parzen Estimator](https://arxiv.org/pdf/2304.11127) algorithm for sequential model-based optimization.
//!

pub mod random_search;
pub use random_search::{BatchRSState, BatchRandomSearch, RSInfo, RandomSearch};

pub mod sha;
pub use sha::{Sha, ShaState};

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
    kernel::Univariate,
    splitter::{LinearSplit, SqrtSplit},
    tpe,
    tpe::Tpe,
    weighter::UniformWeighter,
};
