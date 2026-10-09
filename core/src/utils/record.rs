#[cfg(feature = "spikes")]
use crate::{ElemSpikeCodomain, ElemSpikeConstCodomain, ElemSpikeConstMultiCodomain, ElemSpikeCostCodomain, ElemSpikeCostConstCodomain, ElemSpikeCostConstMultiCodomain, ElemSpikeCostMultiCodomain, ElemSpikeMultiCodomain, SpikeCodomain, SpikeConstCodomain, SpikeConstMultiCodomain, SpikeCostCodomain, SpikeCostConstCodomain, SpikeCostConstMultiCodomain, SpikeCostMultiCodomain, SpikeMultiCodomain};
use crate::{
    BaseSol, Computed, ConstCodomain, ConstMultiCodomain, CostCodomain, CostConstCodomain, CostConstMultiCodomain, 
    CostMultiCodomain, Domain, ElemConstCodomain, ElemConstMultiCodomain, ElemCostCodomain, ElemCostConstCodomain, 
    ElemCostConstMultiCodomain, ElemCostMultiCodomain, ElemMultiCodomain, ElemSingleCodomain, FidelitySol,
    MultiCodomain, Outcome, SingleCodomain, SolId, SolInfo, StepId, Uncomputed, aliases::TypeCodom, variable::var::ToName,
};


pub trait Schema<C> {
    type Parts: Clone;
    fn columns(context: &C, part: Self::Parts) -> Vec<String>;
}

pub trait Recordable<C> {
    type Parts: Clone;
    fn record(&self, context: &C, part: Self::Parts) -> Vec<String>;
}

impl<Out:Outcome> Schema<SingleCodomain<Out>> for ElemSingleCodomain{
    type Parts = ();
    fn columns(_context: &SingleCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        vec!["y".into()]
    }
    
}

impl<Out:Outcome> Recordable<SingleCodomain<Out>> for ElemSingleCodomain {
    type Parts = ();
    fn record(&self, _context: &SingleCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        vec![self.value.to_string()]
    }
}

impl<Out: Outcome> Schema<CostCodomain<Out>> for ElemCostCodomain {
    type Parts=();
    fn columns(_context: &CostCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        vec!["y".into(), "cost".into()]
    }
}

impl<Out:Outcome> Recordable<CostCodomain<Out>> for ElemCostCodomain {
    type Parts = ();

    fn record(&self, _context: &CostCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        vec![self.value.to_string(), self.cost.to_string()]
    }
}

impl<Out:Outcome> Schema<ConstCodomain<Out>> for ElemConstCodomain {
    type Parts = ();
    fn columns(context: &ConstCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v = Vec::from([String::from("y")]);
        v.extend(
            context.c_criteria
                .iter()
                .enumerate()
                .map(|(idx, _)| format!("c{}", idx)),
        );
        v
    }
    
}

impl<Out:Outcome> Recordable<ConstCodomain<Out>> for ElemConstCodomain {
    type Parts = ();

    fn record(&self, _context: &ConstCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v = Vec::from([self.value.to_string()]);
        v.extend(self.constraints.iter().map(|c| c.to_string()));
        v
    }
}

impl<Out:Outcome> Schema<CostConstCodomain<Out>> for ElemCostConstCodomain {
    type Parts = ();
    fn columns(context: &CostConstCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v = Vec::from([String::from("y"), String::from("cost")]);
        v.extend(
            context.c_criteria
                .iter()
                .enumerate()
                .map(|(idx, _)| format!("c{}", idx)),
        );
        v
    }
    
}

impl<Out:Outcome> Recordable<CostConstCodomain<Out>> for ElemCostConstCodomain {
    type Parts = ();

    fn record(&self, _context: &CostConstCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v = Vec::from([self.value.to_string(), self.cost.to_string()]);
        v.extend(self.constraints.iter().map(|c| c.to_string()));
        v
    }
}

impl<Out: Outcome> Schema<MultiCodomain<Out>> for ElemMultiCodomain {
    type Parts = ();
    fn columns(context: &MultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        context
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect()
    }
    
}

impl<Out: Outcome> Recordable<MultiCodomain<Out>> for ElemMultiCodomain {
    type Parts = ();

    fn record(&self, _context: &MultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        self.value.iter().map(|v| v.to_string()).collect()
    }
}




impl<Out: Outcome> Schema<CostMultiCodomain<Out>> for ElemCostMultiCodomain {
    type Parts = ();
    fn columns(context: &CostMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = context
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.push(String::from("cost"));
        v
    }
}

impl<Out: Outcome> Recordable<CostMultiCodomain<Out>> for ElemCostMultiCodomain {
    type Parts = ();

    fn record(&self, _context: &CostMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = self.value.iter().map(|val| val.to_string()).collect();
        v.push(self.cost.to_string());
        v
    }
}

impl<Out: Outcome> Schema<ConstMultiCodomain<Out>> for ElemConstMultiCodomain {
    type Parts = ();
    fn columns(context: &ConstMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = context
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        let c: Vec<String> = context
            .c_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("c{}", idx))
            .collect();
        v.extend(c);
        v
    }
}

impl<Out: Outcome> Recordable<ConstMultiCodomain<Out>> for ElemConstMultiCodomain {
    type Parts = ();

    fn record(&self, _context: &ConstMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = self.value.iter().map(|val| val.to_string()).collect();
        let c: Vec<String> = self.constraints.iter().map(|cst| cst.to_string()).collect();
        v.extend(c);
        v
    }
}

impl<Out: Outcome> Schema<CostConstMultiCodomain<Out>> for ElemCostConstMultiCodomain {
    type Parts = ();
    fn columns(context: &CostConstMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = context
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.push(String::from("cost"));
        let c: Vec<String> = context
            .c_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("c{}", idx))
            .collect();
        v.extend(c);
        v
    }
}

impl<Out: Outcome> Recordable<CostConstMultiCodomain<Out>> for ElemCostConstMultiCodomain {
    type Parts = ();
    fn record(&self, _context: &CostConstMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = self.value.iter().map(|val| val.to_string()).collect();
        v.push(self.cost.to_string());
        let c: Vec<String> = self.constraints.iter().map(|cst| cst.to_string()).collect();
        v.extend(c);
        v
    }
}


#[cfg(feature = "spikes")]
impl<Out:Outcome> Recordable<SpikeCodomain<Out>> for ElemSpikeCodomain{
    type Parts = ();
    fn record(&self, _context: &SpikeCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        vec![self.value.to_string(), self.samples.to_string(), self.spiking.to_string()]
    }
}
#[cfg(feature = "spikes")]
impl<Out:Outcome> Schema<SpikeCodomain<Out>> for ElemSpikeCodomain{
    type Parts = ();
    fn columns(_context: &SpikeCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        vec![String::from("y"), String::from("samples"), String::from("spike")]
    }
}

#[cfg(feature = "spikes")]
impl<Out:Outcome> Recordable<SpikeCostCodomain<Out>> for ElemSpikeCostCodomain{
    type Parts = ();
    fn record(&self, _context: &SpikeCostCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        vec![self.value.to_string(), self.cost.to_string(), self.samples.to_string(), self.spiking.to_string()]
    }
}
#[cfg(feature = "spikes")]
impl<Out:Outcome> Schema<SpikeCostCodomain<Out>> for ElemSpikeCostCodomain{
    type Parts = ();
    fn columns(_context: &SpikeCostCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        vec![String::from("y"), String::from("cost"), String::from("samples"), String::from("spike")]
    }
}

#[cfg(feature = "spikes")]
impl<Out:Outcome> Recordable<SpikeConstCodomain<Out>> for ElemSpikeConstCodomain{
    type Parts = ();
    fn record(&self, _context: &SpikeConstCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v = Vec::from([self.value.to_string(), self.samples.to_string(), self.spiking.to_string()]);
        let c: Vec<String> = self.constraints.iter().map(|cst| cst.to_string()).collect();
        v.extend(c);
        v
    }
}
#[cfg(feature = "spikes")]
impl<Out:Outcome> Schema<SpikeConstCodomain<Out>> for ElemSpikeConstCodomain{
    type Parts = ();
    fn columns(context: &SpikeConstCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v = Vec::from([String::from("y"), String::from("samples"), String::from("spike")]);
        let c: Vec<String> = context
            .c_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("c{}", idx))
            .collect();
        v.extend(c);
        v
    }
}

#[cfg(feature = "spikes")]
impl<Out:Outcome> Recordable<SpikeCostConstCodomain<Out>> for ElemSpikeCostConstCodomain{
    type Parts = ();
    fn record(&self, _context: &SpikeCostConstCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v = Vec::from([self.value.to_string(), self.cost.to_string(), self.samples.to_string(), self.spiking.to_string()]);
        let c: Vec<String> = self.constraints.iter().map(|cst| cst.to_string()).collect();
        v.extend(c);
        v
    }
}
#[cfg(feature = "spikes")]
impl<Out:Outcome> Schema<SpikeCostConstCodomain<Out>> for ElemSpikeCostConstCodomain{
    type Parts = ();
    fn columns(context: &SpikeCostConstCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v = Vec::from([String::from("y"), String::from("cost"), String::from("samples"), String::from("spike")]);
        let c: Vec<String> = context
            .c_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("c{}", idx))
            .collect();
        v.extend(c);
        v
    }
}

#[cfg(feature = "spikes")]
impl<Out:Outcome> Recordable<SpikeMultiCodomain<Out>> for ElemSpikeMultiCodomain{
    type Parts = ();
    fn record(&self, _context: &SpikeMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = self.value.iter().map(|val| val.to_string()).collect();
        v.push(self.samples.to_string());
        v.push(self.spiking.to_string());
        v
    }
}
#[cfg(feature = "spikes")]
impl<Out:Outcome> Schema<SpikeMultiCodomain<Out>> for ElemSpikeMultiCodomain{
    type Parts = ();
    fn columns(context: &SpikeMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = context
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.push(String::from("samples"));
        v.push(String::from("spike"));
        v
    }
}

#[cfg(feature = "spikes")]
impl<Out:Outcome> Recordable<SpikeCostMultiCodomain<Out>> for ElemSpikeCostMultiCodomain{
    type Parts = ();
    fn record(&self, _context: &SpikeCostMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = self.value.iter().map(|val| val.to_string()).collect();
        v.push(self.cost.to_string());
        v.push(self.samples.to_string());
        v.push(self.spiking.to_string());
        v
    }
}
#[cfg(feature = "spikes")]
impl<Out:Outcome> Schema<SpikeCostMultiCodomain<Out>> for ElemSpikeCostMultiCodomain{
    type Parts = ();
    fn columns(context: &SpikeCostMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = context
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.push(String::from("cost"));
        v.push(String::from("samples"));
        v.push(String::from("spike"));
        v
    }
}

#[cfg(feature = "spikes")]
impl<Out:Outcome> Recordable<SpikeConstMultiCodomain<Out>> for ElemSpikeConstMultiCodomain{
    type Parts = ();
    fn record(&self, _context: &SpikeConstMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = self.value.iter().map(|val| val.to_string()).collect();
        v.push(self.samples.to_string());
        v.push(self.spiking.to_string());
        let c: Vec<String> = self.constraints.iter().map(|cst| cst.to_string()).collect();
        v.extend(c);
        v
    }
}
#[cfg(feature = "spikes")]
impl<Out:Outcome> Schema<SpikeConstMultiCodomain<Out>> for ElemSpikeConstMultiCodomain{
    type Parts = ();
    fn columns(context: &SpikeConstMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = context
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.push(String::from("samples"));
        v.push(String::from("spike"));
        let c: Vec<String> = context
            .c_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("c{}", idx))
            .collect();
        v.extend(c);
        v
    }
}

#[cfg(feature = "spikes")]
impl<Out:Outcome> Recordable<SpikeCostConstMultiCodomain<Out>> for ElemSpikeCostConstMultiCodomain{
    type Parts = ();
    fn record(&self, _context: &SpikeCostConstMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = self.value.iter().map(|val| val.to_string()).collect();
        v.push(self.cost.to_string());
        v.push(self.samples.to_string());
        v.push(self.spiking.to_string());
        let c: Vec<String> = self.constraints.iter().map(|cst| cst.to_string()).collect();
        v.extend(c);
        v
    }
}
#[cfg(feature = "spikes")]
impl<Out:Outcome> Schema<SpikeCostConstMultiCodomain<Out>> for ElemSpikeCostConstMultiCodomain{
    type Parts = ();
    fn columns(context: &SpikeCostConstMultiCodomain<Out>, _part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = context
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.push(String::from("cost"));
        v.push(String::from("samples"));
        v.push(String::from("spike"));
        let c: Vec<String> = context
            .c_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("c{}", idx))
            .collect();
        v.extend(c);
        v
    }
}




/// Parts of a solution that can be recorded separately.
/// When used with [`Recordable`] or [`Schema`] types, the [`Id`] is always included.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SolParts {
    /// Only the variable values ([`X`](crate::HasX::X))
    X,
    /// Only the solution metadata ([`Info`](crate::HasInfo::Info))
    Info,
    /// All parts of the solution
    All,
    /// The part without the [`Id`], which is always included in the output.
    NoId(Box<Self>),
}

fn no_id_columns<Info, T>(context: &[T],part: &SolParts) -> Vec<String>
where
    T: ToName,
    Info: SolInfo + Schema<(), Parts = ()>,
{
    match part {
        SolParts::X => context.iter().map(|var| var.to_name()).collect(),
        SolParts::Info => Info::columns(&(), ()),
        SolParts::All => {
            let mut part: Vec<String> = context.iter().map(|var| var.to_name()).collect();
            part.extend(Info::columns(&(), ()));
            part
        },
        SolParts::NoId(inner) => no_id_columns::<Info, T>(context, inner),
    }
}

fn fid_no_id_columns<Info, T>(context: &[T], part: &SolParts) -> Vec<String> 
where
    T: ToName,
    Info: SolInfo + Schema<(), Parts = ()>,
{
    match part {
        SolParts::X => context.iter().map(|var| var.to_name()).collect(),
        SolParts::Info => {
            let mut fid = vec![String::from("fidelity"), String::from("step")];
            let info = Info::columns(&(), ());
            fid.extend(info);
            fid
        },
        SolParts::All => {
            let mut part: Vec<String> = context.iter().map(|var| var.to_name()).collect();
            let fid = vec![String::from("fidelity"), String::from("step")];
            let info = Info::columns(&(), ());
            part.extend(fid);
            part.extend(info);
            part
        },
        SolParts::NoId(inner) => fid_no_id_columns::<Info, T>(context, inner),
    }
}

impl<Id, Info, T, Dom> Schema<&[T]> for BaseSol<Id, Dom,Info> 
where
    Dom: Domain,
    T: ToName,
    Id: SolId + Schema<(), Parts = ()>,
    Info: SolInfo + Schema<(), Parts = ()>,
{
    type Parts = SolParts;
    fn columns(context: &&[T], part: Self::Parts) -> Vec<String> {
        let sol_part = match part {
            SolParts::NoId(sol_parts) => return no_id_columns::<Info, T>(context, &sol_parts),
            _ => no_id_columns::<Info, T>(context, &part),
        };

        let mut id = Id::columns(&(), ());
        id.extend(sol_part);
        id
    }
}

impl<Id, Info, T, Dom> Schema<&[T]> for FidelitySol<Id, Dom,Info> 
where
    Dom: Domain,
    T: ToName,
    Id: StepId + Schema<(), Parts = ()>,
    Info: SolInfo + Schema<(), Parts = ()>,
{
    type Parts = SolParts;
    fn columns(context: &&[T], part: Self::Parts) -> Vec<String> {
        let sol_part = match part {
            SolParts::NoId(sol_parts) => return fid_no_id_columns::<Info, T>(context, &sol_parts),
            _ => fid_no_id_columns::<Info, T>(context, &part),
        };

        let mut id = Id::columns(&(), ());
        id.extend(sol_part);
        id
    }
}

impl<Id, Info, Dom> Recordable<()> for BaseSol<Id, Dom,Info> 
where 
    Dom: Domain,
    Id: SolId + Recordable<(), Parts = ()>,
    Info: SolInfo + Recordable<(), Parts = ()>,
{
    type Parts = SolParts;

    fn record(&self, _context: &(), part: Self::Parts) -> Vec<String> {
        let sol_part = match part {
            SolParts::X => self.x.iter().map(|v| v.to_string()).collect(),
            SolParts::Info => self.info.record(&(), ()),
            SolParts::All => {
                let mut x: Vec<String> = self.x.iter().map(|v| v.to_string()).collect();
                let info = self.info.record(&(), ());
                x.extend(info);
                x
            },
            SolParts::NoId(sol_parts) => {
                return match *sol_parts {
                    SolParts::X => self.x.iter().map(|v| v.to_string()).collect(),
                    SolParts::Info => self.info.record(&(), ()),
                    SolParts::All => {
                        let mut x: Vec<String> = self.x.iter().map(|v| v.to_string()).collect();
                        let info = self.info.record(&(), ());
                        x.extend(info);
                        x
                    },
                    SolParts::NoId(in_part) => self.record(&(), SolParts::NoId(in_part)),
                }
            },
        };
        let mut id = self.id.record(&(), ());
        id.extend(sol_part);
        id
    }
}

impl<Id, Info, Dom> Recordable<()> for FidelitySol<Id, Dom,Info> 
where 
    Dom: Domain,
    Id: StepId + Recordable<(), Parts = ()>,
    Info: SolInfo + Recordable<(), Parts = ()>,
{
    type Parts = SolParts;

    fn record(&self, _context: &(), part: Self::Parts) -> Vec<String> {
        let sol_part = match part {
            SolParts::X => self.x.iter().map(|v| v.to_string()).collect(),
            SolParts::Info => {
                let mut fid = vec![self.fid.to_string(), self.step.to_string()];
                let info = self.info.record(&(), ());
                fid.extend(info);
                fid
            },
            SolParts::All => {
                let mut x: Vec<String> = self.x.iter().map(|v| v.to_string()).collect();
                let info = {
                    let mut fid = vec![self.fid.to_string(), self.step.to_string()];
                    let info = self.info.record(&(), ());
                    fid.extend(info);
                    fid
                };
                x.extend(info);
                x
            },
            SolParts::NoId(sol_parts) => {
                return match *sol_parts {
                    SolParts::X => self.x.iter().map(|v| v.to_string()).collect(),
                    SolParts::Info => {
                        let mut fid = vec![self.fid.to_string(), self.step.to_string()];
                        let info = self.info.record(&(), ());
                        fid.extend(info);
                        fid
                    },
                    SolParts::All => {
                        let mut x: Vec<String> = self.x.iter().map(|v| v.to_string()).collect();
                        let info = {
                            let mut fid = vec![self.fid.to_string(), self.step.to_string()];
                            let info = self.info.record(&(), ());
                            fid.extend(info);
                            fid
                        };
                        x.extend(info);
                        x
                    },
                    SolParts::NoId(in_part) => self.record(&(), SolParts::NoId(in_part)),
                }
            },
        };
        let mut id = self.id.record(&(), ());
        id.extend(sol_part);
        id
    }
}

/// Parts of a [`Computed`] solution that can be recorded separately.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompParts {
    /// Select a part of the computed solution to record.
    Sol(SolParts),
    /// Only the ([`Y`](crate::HasY::Y))
    Y,
    /// All parts of the solution
    All,
    /// The part without the [`Id`], which is always included in the output.
    NoId(Box<Self>),
}

impl<'a, S, Out, T, Dom> Schema<(&'a [T], &'a Out::Cod)> for Computed<S, Out>
where 
    S: Uncomputed<Dom = Dom> + Schema<&'a [T], Parts = SolParts>,
    Out: Outcome,
    Dom: Domain,
{
    type Parts = CompParts;
    fn columns(context: &(&'a [T], &'a Out::Cod), part: Self::Parts) -> Vec<String> {
        let sol_part = match part {
            CompParts::Sol(sol_parts) => S::columns(&context.0, SolParts::NoId(sol_parts.into())),
            CompParts::Y => TypeCodom::<Out>::columns(context.1, ()),
            CompParts::All => {
                let mut sol_cols = S::columns(&context.0, SolParts::NoId(SolParts::All.into()));
                let y_cols = TypeCodom::<Out>::columns(context.1, ());
                sol_cols.extend(y_cols);
                sol_cols
            },
            CompParts::NoId(comp_parts) => {
                return match *comp_parts {
                    CompParts::Sol(sol_parts) => S::columns(&context.0, SolParts::NoId(sol_parts.into())),
                    CompParts::Y => TypeCodom::<Out>::columns(context.1, ()),
                    CompParts::All => {
                        let mut sol_cols = S::columns(&context.0, SolParts::NoId(SolParts::All.into()));
                        let y_cols = TypeCodom::<Out>::columns(context.1, ());
                        sol_cols.extend(y_cols);
                        sol_cols
                    },
                    CompParts::NoId(in_part) => Self::columns(context, CompParts::NoId(in_part)),
                }
            }
        };
        let mut id = S::Id::columns(&(), ());
        id.extend(sol_part);
        id
    }
}

impl<S, Out, Dom> Recordable<Out::Cod> for Computed<S, Out>
where 
    S: Uncomputed<Dom = Dom> + Recordable<(), Parts = SolParts>,
    Out: Outcome,
    Dom: Domain,
{
    type Parts = CompParts;

    fn record(&self, context: &Out::Cod, part: Self::Parts) -> Vec<String> {
        let sol_part =match part {
            CompParts::Sol(sol_parts) => self.sol.record(&(), SolParts::NoId(sol_parts.into())),
            CompParts::Y => self.y.record(context, ()),
            CompParts::All => {
                let mut sol_rec = self.sol.record(&(), SolParts::NoId(SolParts::All.into()));
                let y_rec = self.y.record(context, ());
                sol_rec.extend(y_rec);
                sol_rec
            },
            CompParts::NoId(comp_parts) => {
                return match *comp_parts {
                    CompParts::Sol(sol_parts) => self.sol.record(&(), SolParts::NoId(sol_parts.into())),
                    CompParts::Y => self.y.record(context, ()),
                    CompParts::All => {
                        let mut sol_rec = self.sol.record(&(), SolParts::NoId(SolParts::All.into()));
                        let y_rec = self.y.record(context, ());
                        sol_rec.extend(y_rec);
                        sol_rec
                    },
                    CompParts::NoId(in_part) => self.record(context, CompParts::NoId(in_part)),
                }
            }
        };
        let mut id = self.sol.id().record(&(), ());
        id.extend(sol_part);
        id
    }
}