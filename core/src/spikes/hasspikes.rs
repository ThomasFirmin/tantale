/// Trait for objects containing a number of spiking samples
///
/// [`HasSpikes`] tracks the number of spiking samples in a solution.
pub trait HasSpikes {
    /// Returns the total number of samples in the solution.
    fn samples(&self) -> usize;
    /// Returns the number of spiking samples in the solution.
    fn spiking_samples(&self) -> usize;
    /// Returns the number of non-spiking samples in the solution.
    fn non_spiking_samples(&self) -> usize{
        self.samples() - self.spiking_samples()
    }
}