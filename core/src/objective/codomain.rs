use crate::objective::outcome::Outcome;
use crate::objective::criteria::Criteria;

/// This trait defines what a [`Codomain`] is, i.e. the output of the [`Objective`](tantale::core::objective::Objective) function.
pub trait Codomain{}

/// Defines a mono-objective [`Codomain`], i.e. $f(x)=y$
pub trait Single<Out, C> : Codomain
where
    Out : Outcome,
    C : Criteria,
{
    fn get_criteria(&self) -> C;
    fn get_y(&self, o:Out)-> f64{
        self.get_criteria().extract(o)
    }
}

/// Defines a mono-objective [`Codomain`], i.e. $F(x)=f_1(x),f_2(x),\dots,f_k(x)$
pub trait Multi<Out> : Codomain
where
    Out : Outcome,
{
    fn get_criteria(&self) -> Vec<Box<dyn Criteria>>;
    fn get_y(&self, o:Out)-> Box<[f64]>{
        let criterias = self.get_criteria();
        criterias.iter().map(|c | c.extract(o)).collect()
    }
}

/// Defines a [`Codomain`] constrained by equalities or inequalities depending on [`ConsType`].
pub trait Constrained<Out, ConsType> : Codomain
where
    Out : Outcome,
{
    fn get_criteria(&self) -> Vec<Box<dyn Criteria>>;
    fn get_constraints(&self, o:Out)-> Box<[f64]>{
        self.get_criteria().iter().map(|c | c.extract(o)).collect()
    }
}

/// Type of constraints for a [`Constrained`] [`Codomain`].
/// * [`Equality`](ConsType::Equality) : $f_c_i(x) \leq c_i ,\text{ for }i = 0,\dots,k$, where $c_i$ is the constraint  $f_c_i(x)$ is the evaluation of the constraint $i$ by the [`Objective`].
///   In practice,  the constraint is expressed as $f_c_i(x) - c_i \leq 0  ,\text{ for }i = 0,\dots,k$.
/// * [`Inequality`](ConsType::Inequality) : $f_c_i(x) = c_i,\text{ for }i = 0,\dots,k$, where $c_i$ is the constraint  $f_c(x)$ is the evaluation of the constraint $i$ by the [`Objective`].
///   In practice,  the constraint is expressed as $f_c_i(x) - c_i = 0 ,\text{ for }i = 0,\dots,k$.
/// * [`Both`](ConsType::Both) : Constraints can be both equalities or inequalities.
pub enum ConsType {
    Equality,
    Inequality,
    Both,
}


pub trait Fidelity<Out,C> : Codomain
where
    Out : Outcome,
    C : Criteria,
{
    fn get_criteria(&self) -> C;
    fn get_fidelity(&self, o:Out)-> f64{
        self.get_criteria().extract(o)
    }
}



// MONO OBJECTIVE CODOMAINS

/// A single [`Criteria`] [`Codomain`] made of a single value `y`.
pub struct SingleCodomain<C:Criteria>
{
    pub criteria : C,
    pub y : f64,
}

impl <C:Criteria> Codomain for SingleCodomain<C>{}

impl <Out,C> Single<Out,C> for SingleCodomain<C>
where
    Out : Outcome,
    C : Criteria,
{
    fn get_criteria(&self) -> C {
        self.criteria
    }
}

/// A [`Single`] and [`Fidelity`] [`Codomain`], i.e. $f(x) = y, \tau $, with $\tau$ the fidelity.
pub struct FidelCodomain<C:Criteria>
{
    pub criteria : C,
    pub y : f64,
    pub fidelity : f64,
}
impl <C:Criteria> Codomain for FidelCodomain<C>{}

impl <Out,C> Single<Out,C> for FidelCodomain<C>
where
    Out : Outcome,
    C : Criteria,
{
    fn get_criteria(&self) -> C {
        self.criteria
    }
}
impl <Out,C> Fidelity<Out,C> for FidelCodomain<C>
where
    Out : Outcome,
    C : Criteria,
{
    fn get_criteria(&self) -> C {
        self.criteria
    }
}

/// A [`Single`], [`Constrained`], and [`Fidelity`] [`Codomain`], i.e. $f(x) = y, \tau $, with $\tau$ the fidelity.
pub struct ConstrainedCodomain<C:Criteria>
{
    pub y_criteria : C,
    pub y : f64,
    pub c_criteria : Vec<Box<dyn Criteria>>,
    pub constraints : Box<[f64]>,
}

impl <C:Criteria> Codomain for ConstrainedCodomain<C>{}
impl <Out,C> Single<Out,C> for ConstrainedCodomain<C>
where
    Out : Outcome,
    C : Criteria,
{
    fn get_criteria(&self) -> C {
        self.y_criteria
    }
}
impl <Out, C> Constrained<Out,ConsType> for ConstrainedCodomain<C>
where
    Out : Outcome,
    C : Criteria,
{
    fn get_criteria(&self) -> Vec<Box<dyn Criteria>> {
        self.c_criteria
    }
}


pub struct FidelConstCodomain<Out>
where
    Out : Outcome,
{

}
impl Codomain for FidelConstCodomain{

}
impl Single for FidelConstCodomain{

}
impl Fidelity for FidelConstCodomain{

}
impl Constrained for FidelConstCodomain{

}

pub type ConstFidelCodomain = FidelConstCodomain;


// MULTI OBJECTIVE CODOMAINS

pub struct MultiCodomain<Out>
where
    Out : Outcome,
{}
impl Codomain for MultiCodomain{

}

pub struct FidelMultiCodomain<Out>
where
    Out : Outcome,
{}
impl Codomain for FidelMultiCodomain{

}
impl Multi for MultiCodomain{

}
impl Fidelity for FidelMultiCodomain {
    
}

pub struct ConstMultiCodomain<Out>
where
    Out : Outcome,
{}
impl Codomain for ConstMultiCodomain{

}
impl Multi for ConstMultiCodomain{

}
impl Constrained for ConstMultiCodomain{

}

pub struct FidelConstMultiCodomain<Out>
where
    Out : Outcome,
{}
impl Codomain for FidelConstMultiCodomain{

}
impl Multi for FidelConstMultiCodomain{

}
impl Constrained for FidelConstMultiCodomain{
    
}

pub type ConstFidelMultiCodomain = FidelConstMultiCodomain;