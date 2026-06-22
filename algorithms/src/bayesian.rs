//! Bayesian optimization algorithms for hyperparameter tuning.

pub mod tpe;
pub use tpe::Tpe;

pub mod splitter;
pub use splitter::{LinearSplit, SqrtSplit, MOSplit, Splitter};

pub mod weighter;
pub use weighter::{Weighter, PointWeights};

pub mod kernel;
pub use kernel::{AitchisonAitkenKernel, GaussianKernel, MixedKernel, Univariate, Multivariate};

pub mod bandwidth;
pub use bandwidth::{optuna_bw, magic_clip};

pub mod error;

