//! Bayesian optimization algorithms for hyperparameter tuning.

pub mod tpe;
pub use tpe::Tpe;

pub mod splitter;
pub use splitter::{LinearSplit, MOSplit, Splitter, SqrtSplit};

pub mod weighter;
pub use weighter::{PointWeights, Weighter};

pub mod kernel;
pub use kernel::{AitchisonAitkenKernel, GaussianKernel, MixedKernel, Multivariate, Univariate};

pub mod bandwidth;
pub use bandwidth::{magic_clip, optuna_bw};

pub mod error;
