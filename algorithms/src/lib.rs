//! This crate contains some algorithms implemented in Tantale,
//! using the core library.
//! 
//! The following algorithms are implemented:
//! - [`random_search`] :
//!   * [`RandomSearch`] : the usual sequential [Random Search](https://en.wikipedia.org/wiki/Random_search) algorithm. Generating points on demand. 
//!   * [`BatchRandomSearch`] : a batch version of random search. Generating points in batches.

pub mod random_search;
pub use random_search::{BatchRSState, BatchRandomSearch, RSInfo, RandomSearch};
