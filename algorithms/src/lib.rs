//! This crate contains some algorithms implemented in Tantale,
//! using the core library.
//!
//! The following algorithms are implemented:
//! - [`Random Search`](random_search) :
//!   * [`RandomSearch`] : the usual sequential [Random Search](https://en.wikipedia.org/wiki/Random_search) algorithm. Generating points on demand.
//!   * [`BatchRandomSearch`] : a batch version of random search. Generating points in batches.
//! - [`Successive Halving`](sha) :
//!   * [`Sha`] : the original [Successive Halving](https://arxiv.org/pdf/1502.07943) algorithm for multi-fidelity hyperparameter optimization.
//! - [`Asynchronous Successive Halving`](asha) :
//!   * [`Asha`] : the asynchronous version of [Successive Halving](https://arxiv.org/pdf/1810.05934) algorithm for multi-fidelity hyperparameter optimization.

pub mod random_search;
pub use random_search::{BatchRSState, BatchRandomSearch, RSInfo, RandomSearch};

pub mod sha;
pub use sha::{ShaInfo, ShaState, Sha};

pub mod asha;
pub use asha::Asha;

pub mod hyperband;
pub use hyperband::Hyperband;