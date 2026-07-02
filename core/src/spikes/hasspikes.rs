use crate::{Computed, Domain, ElemSpikeCodomain, ElemSpikeConstCodomain, ElemSpikeConstMultiCodomain, ElemSpikeCostCodomain, ElemSpikeCostConstCodomain, ElemSpikeCostConstMultiCodomain, ElemSpikeCostMultiCodomain, ElemSpikeMultiCodomain, HasY, Id, Outcome, SolInfo, TypeCodom, Uncomputed, solution::{CompLone, shape::CompPair}};

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


impl HasSpikes for ElemSpikeCodomain{
    fn samples(&self) -> usize {
        self.samples
    }

    fn spiking_samples(&self) -> usize {
        self.spiking
    }
}
impl HasSpikes for ElemSpikeCostCodomain{
    fn samples(&self) -> usize {
        self.samples
    }
    fn spiking_samples(&self) -> usize {
        self.spiking
    }
    
}
impl HasSpikes for ElemSpikeConstCodomain{
    fn samples(&self) -> usize {
        self.samples
    }
    fn spiking_samples(&self) -> usize {
        self.spiking
    }
    
}
impl HasSpikes for ElemSpikeCostConstCodomain{
    fn samples(&self) -> usize {
        self.samples
    }
    fn spiking_samples(&self) -> usize {
        self.spiking
    }
    
}
impl HasSpikes for ElemSpikeMultiCodomain{
    fn samples(&self) -> usize {
        self.samples
    }
    fn spiking_samples(&self) -> usize {
        self.spiking
    }
    
}
impl HasSpikes for ElemSpikeCostMultiCodomain{
    fn samples(&self) -> usize {
        self.samples
    }
    fn spiking_samples(&self) -> usize {
        self.spiking
    }
    
}
impl HasSpikes for ElemSpikeConstMultiCodomain{
    fn samples(&self) -> usize {
        self.samples
    }
    fn spiking_samples(&self) -> usize {
        self.spiking
    }
    
}
impl HasSpikes for ElemSpikeCostConstMultiCodomain{
    fn samples(&self) -> usize {
        self.samples
    }
    fn spiking_samples(&self) -> usize {
        self.spiking
    }
    
}

impl<PSol, SolId, Dom, SInfo, Out> HasSpikes for Computed<PSol, SolId, Dom, Out, SInfo>
where
    Self: HasY<Out>,
    TypeCodom<Out>: HasSpikes,
    Out: Outcome,
    PSol: Uncomputed<SolId, Dom, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Dom: Domain,
{
    fn samples(&self) -> usize {
        self.y.samples()
    }

    fn spiking_samples(&self) -> usize {
        self.y.spiking_samples()
    }
}

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo, Out> HasSpikes
    for CompPair<SolObj, SolOpt, SolId, Obj, Opt, SInfo, Out>
where
    Self: HasY<Out>,
    TypeCodom<Out>: HasSpikes,
    Out: Outcome,
    SolObj: Uncomputed<SolId, Obj, SInfo>,
    SolOpt: Uncomputed<SolId, Opt, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Obj: Domain,
    Opt: Domain,
{
    fn samples(&self) -> usize {
        self.y().samples()
    }

    fn spiking_samples(&self) -> usize {
        self.y().spiking_samples()
    }
}

impl<SolObj, SolId, Obj, SInfo, Out> HasSpikes for CompLone<SolObj, SolId, Obj, SInfo, Out>
where
    Self: HasY<Out>,
    TypeCodom<Out>: HasSpikes,
    Out: Outcome,
    SolObj: Uncomputed<SolId, Obj, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Obj: Domain,
{
    fn samples(&self) -> usize {
        self.y().samples()
    }

    fn spiking_samples(&self) -> usize {
        self.y().spiking_samples()
    }
}
