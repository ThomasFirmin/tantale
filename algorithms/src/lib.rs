//! This crate contains some algorithms implemented in Tantale,
//! using the core library.
//!
//! The following algorithms are implemented:
//! - [`random_search`] :
//!   * [`RandomSearch`] : the usual sequential [Random Search](https://en.wikipedia.org/wiki/Random_search) algorithm. Generating points on demand.
//!   * [`BatchRandomSearch`] : a batch version of random search. Generating points in batches.
//! - [`successive_halving`] :
//!   * [`SuccessiveHalving`] : the original [Successive Halving](https://arxiv.org/pdf/1502.07943) algorithm for multi-fidelity hyperparameter optimization.
//! - [`asha`] :
//!   * [`ASHA`] : the asynchronous version of [Successive Halving](https://arxiv.org/pdf/1810.05934) algorithm for multi-fidelity hyperparameter optimization.

pub mod random_search;
pub use random_search::{BatchRSState, BatchRandomSearch, RSInfo, RandomSearch};

pub mod successive_halving;
pub use successive_halving::{SHInfo, SHState, SuccessiveHalving};

pub mod asha;
pub use asha::ASHA;