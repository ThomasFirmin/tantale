use crate::{
    BaseSol, ConstCodomain, ConstMultiCodomain, CostCodomain, CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain, Domain, ElemConstCodomain, ElemConstMultiCodomain, ElemCostCodomain, ElemCostConstCodomain, ElemCostConstMultiCodomain, ElemCostMultiCodomain, ElemMultiCodomain, ElemSingleCodomain, FidelitySol, HasVariables, Id, MultiCodomain, OntoDom, Outcome, Runable, SingleCodomain, SolInfo, SolutionShape, Sp, Var, domain::PreDomain, recorder::csv::SolCSVWrite,
};

pub trait Schema<C> {
    fn columns(context: &C) -> Vec<String>;
}

pub trait Recordable {
    type Parts;
    fn record(&self, part: Self::Parts) -> Vec<String>;
}

impl<Out:Outcome> Schema<SingleCodomain<Out>> for ElemSingleCodomain{   
    fn columns(context: &SingleCodomain<Out>) -> Vec<String> {
        vec!["y".into()]
    }
    
}

impl Recordable for ElemSingleCodomain {
    type Parts = ();

    fn record(&self, part: Self::Parts) -> Vec<String> {
        vec![self.value.to_string()]
    }
}

impl<Out: Outcome> Schema<CostCodomain<Out>> for ElemCostCodomain {
    fn columns(context: &CostCodomain<Out>) -> Vec<String> {
        vec!["y".into(), "cost".into()]
    }
}

impl Recordable for ElemCostCodomain {
    type Parts = ();

    fn record(&self, part: Self::Parts) -> Vec<String> {
        vec![self.value.to_string(), self.cost.to_string()]
    }
}

impl<Out:Outcome> Schema<ConstCodomain<Out>> for ElemConstCodomain {
    fn columns(context: &ConstCodomain<Out>) -> Vec<String> {
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

impl Recordable for ElemConstCodomain {
    type Parts = ();

    fn record(&self, part: Self::Parts) -> Vec<String> {
        let mut v = Vec::from([self.value.to_string()]);
        v.extend(self.constraints.iter().map(|c| c.to_string()));
        v
    }
}

impl<Out:Outcome> Schema<CostConstCodomain<Out>> for ElemCostConstCodomain {
    fn columns(context: &CostConstCodomain<Out>) -> Vec<String> {
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

impl Recordable for ElemCostConstCodomain {
    type Parts = ();

    fn record(&self, part: Self::Parts) -> Vec<String> {
        let mut v = Vec::from([self.value.to_string(), self.cost.to_string()]);
        v.extend(self.constraints.iter().map(|c| c.to_string()));
        v
    }
}

impl<Out: Outcome> Schema<MultiCodomain<Out>> for ElemMultiCodomain {
    fn columns(context: &MultiCodomain<Out>) -> Vec<String> {
        context
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect()
    }
}

impl Recordable for ElemMultiCodomain {
    type Parts = ();

    fn record(&self, part: Self::Parts) -> Vec<String> {
        self.value.iter().map(|v| v.to_string()).collect()
    }
}




impl<Out: Outcome> Schema<CostMultiCodomain<Out>> for ElemCostMultiCodomain {
    fn columns(context: &CostMultiCodomain<Out>) -> Vec<String> {
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

impl Recordable for ElemCostMultiCodomain {
    type Parts = ();

    fn record(&self, part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = self.value.iter().map(|val| val.to_string()).collect();
        v.push(self.cost.to_string());
        v
    }
}

impl<Out: Outcome> Schema<ConstMultiCodomain<Out>> for ElemConstMultiCodomain {
    fn columns(context: &ConstMultiCodomain<Out>) -> Vec<String> {
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

impl Recordable for ElemConstMultiCodomain {
    type Parts = ();

    fn record(&self, part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = self.value.iter().map(|val| val.to_string()).collect();
        let c: Vec<String> = self.constraints.iter().map(|cst| cst.to_string()).collect();
        v.extend(c);
        v
    }
}

impl<Out: Outcome> Schema<CostConstMultiCodomain<Out>> for ElemCostConstMultiCodomain {
    fn columns(context: &CostConstMultiCodomain<Out>) -> Vec<String> {
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

impl Recordable for ElemCostConstMultiCodomain {
    type Parts = ();

    fn record(&self, part: Self::Parts) -> Vec<String> {
        let mut v: Vec<String> = self.value.iter().map(|val| val.to_string()).collect();
        v.push(self.cost.to_string());
        let c: Vec<String> = self.constraints.iter().map(|cst| cst.to_string()).collect();
        v.extend(c);
        v
    }
}



/// Parts of a solution that can be recorded separately.
pub enum SolParts {
    /// Only the variable values ([`X`](crate::HasX::X))
    X,
    /// Only the solution identifier ([`Id`](crate::HasId::Id))
    Id,
    /// Only the solution metadata ([`Info`](crate::HasInfo::Info))
    Info,
    /// All parts of the solution
    All,
}

/// Parts of a [`Computed`] solution that can be recorded separately.
pub enum CompParts {
    /// Select a part of the computed solution to record.
    Sol(SolParts),
    /// Only the ([`Y`](crate::HasY::Y))
    Y,
    /// All parts of the solution
    All,
}

/// Parts of a solution shape that can be recorded separately.
pub enum ShapePart {
    /// Only the objective solution ([`Obj`](crate::solution::shape::SolutionShape::ObjSol))
    Obj(SolParts),
    /// Only the optimizer solution ([`Opt`](crate::solution::shape::SolutionShape::OptSol))
    Opt(SolParts),
}

impl<R: Runable> Schema for BaseSol<>
{
    
}

impl<SolId, Dom, Info> Tabular<BaseSol<SolId, Dom, Info>> for Var<>
where
    Dom: Domain,
    SolId: Id + Tabular<(), Parts = ()> + HasHeader,
    Info: SolInfo + Tabular<(), Parts = ()> + HasHeader,
{
    type Parts = SolParts;
    fn column_names(&self, part: Self::Parts) -> Vec<String> {
        let mut id = SolId::header();
        let mut part = match part {
            SolParts::X => self.names(),
            SolParts::Id => Vec::new(),
            SolParts::Info => Info::header(),
            SolParts::All => {
                let mut v = self.names();
                v.extend(Info::header());
                v
            },
        };
        id.extend(part);
        id
    }

    /// Write value as CSV row columns.
    fn column_values(&self, elem: BaseSol<SolId, Dom, Info>, part: Self::Parts) -> Vec<String> {
        let mut id = elem.id.column_values(&(), ());
        let mut part = match part {
            SolParts::X => elem.x.iter().map(|v| v.to_string()).collect(),
            SolParts::Id => Vec::new(),
            SolParts::Info => elem.info.column_values(&(), ()),
            SolParts::All => {
                let mut v = self.values(&elem.x);
                v.extend(elem.info.column_values(&(), ()));
                v
            },
        };
        id.extend(part);
        id
    }
}

impl<H, SolId, Dom, Info> Tabular<FidelitySol<SolId, Dom, Info>> for Vec<String>
where
    H: HasVariables,
    SolId: Id + Tabular<(), Parts = ()> + HasHeader,
    Dom: Domain,
    Info: SolInfo + Tabular<(), Parts = ()> + HasHeader,
{
    type Parts = SolParts;
    fn column_names(&self, part: Self::Parts) -> Vec<String> {
        let mut id = SolId::header();
        let mut part = match part {
            SolParts::X => self.names(),
            SolParts::Id => Vec::new(),
            SolParts::Info => Info::header(),
            SolParts::All => {
                let mut v = self.names();
                v.extend(Info::header());
                v
            },
        };
        id.extend(part);
        id
    }

    /// Write value as CSV row columns.
    fn column_values(&self, elem: FidelitySol<SolId, Dom, Info>, part: Self::Parts) -> Vec<String> {
        let mut id = elem.id.column_values(&(), ());
        let mut part = match part {
            SolParts::X => elem.x.iter().map(|v| v.to_string()).collect(),
            SolParts::Id => Vec::new(),
            SolParts::Info => {
                let mut fid = vec![elem.fid.to_string(), elem.step.to_string()];
                let mut info = elem.info.column_values(&(), ());
                fid.extend(info);
                fid
            },
            SolParts::All => {
                let mut v = self.values(&elem.x);
                v.extend(elem.info.column_values(&(), ()));
                v
            },
        };
        id.extend(part);
        id
    }
}

impl<Shape, Obj, Opt> Tabular<Shape> for Sp<Obj, Opt>
where
    Shape : SolutionShape<Obj = Obj, Opt = Opt>,
    Shape::Id: Id + Tabular<(), Parts = ()> + HasHeader,
    Shape::Info: SolInfo + Tabular<(), Parts = ()> + HasHeader,
    Obj: Domain,
    Opt: PreDomain,
{
    type Parts = ShapePart;

    fn column_names(&self, part: Self::Parts) -> Vec<String> {
        match part {
            ShapePart::Obj(sol_parts) => <Self as HasVariables>::,
            ShapePart::Opt(sol_parts) => self.column_names(sol_parts),
        }
    }

    fn column_values(&self, elem: &Shape, part: Self::Parts) -> Vec<String> {
        match part {
            ShapePart::Obj(sol_parts) => self.column_values(elem.get_sobj(), sol_parts),
            ShapePart::Opt(sol_parts) => self.column_values(elem.get_sopt(), sol_parts),
        }
    }
}

impl<SolId, Out> HasHeader for (SolId, Out)
where
    SolId: Id + Tabular<(), Parts = ()> + HasHeader,
    Out: SolInfo + Tabular<(), Parts = ()> + HasHeader,
{
    fn header() -> Vec<String> {
        let mut id = SolId::header();
        let mut out = Out::header();
        id.extend(out);
        id
    }
}

impl<SolId, Out> Tabular<()> for (SolId, Out)
where
    SolId: Id + Tabular<(), Parts = ()> + HasHeader,
    Out: SolInfo + Tabular<(), Parts = ()> + HasHeader,
{
    type Parts = ();
    fn column_names(&self, part: Self::Parts) -> Vec<String> {
        Self::header()
    }

    fn column_values(&self, elem: &(), part: Self::Parts) -> Vec<String> {
        let (sol_id, out) = self;
        let mut id = sol_id.column_values(&(), ());
        let mut out = out.column_values(&(), ());
        id.extend(out);
        id
    }
}