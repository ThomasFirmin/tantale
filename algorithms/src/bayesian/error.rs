use std::fmt;
use tantale_core::{err_msg, errors::ErrMsg};

/// Error type for splitting operations from a [`Splitter`](crate::bayesian::splitter::Splitter).
pub enum SplitError{
    /// Error indicating that there are not enough points in the archive to perform a split.
    NotEnoughPoints(&'static str),
    /// Error indicating that the configuration of the splitter is invalid.
    ConfigError(&'static str),
}

impl ErrMsg for SplitError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SplitError::NotEnoughPoints(msg) => write!(f, "Split error: Not enough points in the archive to perform a split. {msg}"),
            SplitError::ConfigError(msg) => write!(f, "Split error: Invalid configuration for the splitter. {msg}"),
        }
    }
}
err_msg!(SplitError);

/// Error type for bayesian prior operations.
pub enum PriorError{
    /// Error indicating that the configuration of the prior is invalid.
    ConfigError(&'static str),
}

impl ErrMsg for PriorError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PriorError::ConfigError(msg) => write!(f, "Prior error: Invalid configuration for the prior. {msg}"),
        }
    }
}
err_msg!(PriorError);

pub enum KernelError{
    /// Error indicating that the configuration of the kernel is invalid.
    ConfigError(&'static str),
    /// Error indicating that there are not enough points in the archive to perform kernel operations.
    NotEnoughPoints(&'static str),
}

impl ErrMsg for KernelError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KernelError::ConfigError(msg) => write!(f, "Kernel error: Invalid configuration for the kernel. {msg}"),
            KernelError::NotEnoughPoints(msg) => write!(f, "Kernel error: Not enough points in the archive to perform kernel operations. {msg}"),
        }
    }
}
err_msg!(KernelError);
pub enum BandwidthError{
    /// Error indicating that the configuration of the bandwidth is invalid.
    ConfigError(&'static str),
    /// Error indicating that the domain type is not supported for bandwidth estimation.
    DomainTypeError(&'static str),
    /// Error indicating that there are not enough points in the archive to perform bandwidth estimation.
    NotEnoughPoints(&'static str),
}

impl ErrMsg for BandwidthError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BandwidthError::ConfigError(msg) => write!(f, "Bandwidth error: Invalid configuration for the bandwidth. {msg}"),
            BandwidthError::DomainTypeError(msg) => write!(f, "Bandwidth error: Unsupported domain type for bandwidth estimation. {msg}"),
            BandwidthError::NotEnoughPoints(msg) => write!(f, "Bandwidth error: Not enough points in the archive to perform bandwidth estimation. {msg}"),
        }
    }
}
err_msg!(BandwidthError);

pub enum WeighterError{
    /// Error indicating that the configuration of the weighter is invalid.
    ConfigError(&'static str),
    /// Error indicating that there are not enough points in the archive to perform weighting.
    NotEnoughPoints(&'static str),
}

impl ErrMsg for WeighterError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WeighterError::ConfigError(msg) => write!(f, "Weighter error: Invalid configuration for the weighter. {msg}"),
            WeighterError::NotEnoughPoints(msg) => write!(f, "Weighter error: Not enough points in the archive to perform weighting. {msg}"),
        }
    }
}
err_msg!(WeighterError);