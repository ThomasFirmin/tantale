use crate::objective::outcome::Outcome;

/// A criteria defines a function taking the [`Outcome`] of the evaluation of the [`Objective`] function
pub type Criteria<Out> = fn(&Out)->f64;

/// This trait defines what a [`Codomain`] is, i.e. the output of the [`Objective`](tantale::core::objective::Objective) function.
/// It has an associated type [`TypeCodom`](Codomain::TypeCodom), defining what an element from the [`Codomain`] is.
pub trait Codomain<Out:Outcome>{
    type TypeCodom;
    fn get_elem(&self, o:&Out)-> Self::TypeCodom;
}

/// Defines a mono-objective [`Codomain`], i.e. $f(x)=y$
pub trait Single<Out:Outcome> : Codomain<Out>
{
    fn get_criteria(&self) -> Criteria<Out>;
    fn get_y(&self, o:&Out)-> f64{
        (self.get_criteria())(o)
    }
}

/// Defines a mono-objective [`Codomain`], i.e. $F(x)=f_1(x),f_2(x),\dots,f_k(x)$
pub trait Multi<Out:Outcome> : Codomain<Out>
{
    fn get_criteria(&self) -> &Box<[Criteria<Out>]>;
    fn get_y(&self, o:&Out)-> Box<[f64]>{
        let criterias = self.get_criteria();
        criterias.iter().map(|c | c(o)).collect()
    }
}

/// Defines a [`Codomain`] constrained by equalities or inequalities depending on [`ConsType`].
pub trait Constrained<Out:Outcome, ConsType> : Codomain<Out>
{
    fn get_criteria(&self) -> &Box<[Criteria<Out>]>;
    fn get_constraints(&self, o:&Out)-> Box<[f64]>{
        self.get_criteria().iter().map(|c | c(o)).collect()
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

pub trait Fidelity<Out:Outcome> : Codomain<Out>
{
    fn get_criteria(&self) -> Criteria<Out>;
    fn get_fidelity(&self, o:&Out)-> f64{
        (self.get_criteria())(o)
    }
}


// MONO OBJECTIVE CODOMAINS

/// A single [`Criteria`] [`Codomain`] made of a single value `y`.
pub struct SingleCodomain<Out:Outcome>
{
    pub y_criteria : Criteria<Out>,
}

impl <Out:Outcome> Codomain<Out> for SingleCodomain<Out>{
    type TypeCodom = f64;
    
    fn get_elem(&self, o:&Out)-> Self::TypeCodom {
        self.get_y(o)
    }   
}

impl <Out:Outcome> Single<Out> for SingleCodomain<Out>
{
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}

/// A [`Single`] and [`Fidelity`] [`Codomain`].
pub struct FidelCodomain<Out:Outcome>
{
    pub y_criteria : Criteria<Out>,
    pub f_criteria : Criteria<Out>,
}
/// A element ([`TypeCodom`](Codomain::TypeDom)) from [`FidelCodomain`].
pub struct ElemFidelCodomain
{
    pub y : f64,
    pub fidelity : f64,
}

impl <Out:Outcome> Codomain<Out> for FidelCodomain<Out>{
    type TypeCodom = ElemFidelCodomain;

    fn get_elem(&self, o:&Out)-> Self::TypeCodom {
        ElemFidelCodomain{
            y: self.get_y(o),
            fidelity: self.get_fidelity(o),
        }
    }
}

impl <Out:Outcome> Single<Out> for FidelCodomain<Out>
{
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}
impl <Out:Outcome> Fidelity<Out> for FidelCodomain<Out>
{
    fn get_criteria(&self) -> Criteria<Out> {
        self.f_criteria
    }
}

/// A [`Single`] and [`Constrained`] [`Codomain`].
pub struct ConstrainedCodomain<Out:Outcome>
{
    pub y_criteria : Criteria<Out>,
    pub c_criteria : Box<[Criteria<Out>]>,
}
/// A element ([`TypeCodom`](Codomain::TypeDom)) from [`ConstrainedCodomain`]
pub struct ElemConstrainedCodomain
{
    pub y : f64,
    pub constraints : Box<[f64]>,
}

impl <Out:Outcome> Codomain<Out> for ConstrainedCodomain<Out>{
    type TypeCodom=ElemConstrainedCodomain;
    fn get_elem(&self, o:&Out)-> Self::TypeCodom {
        ElemConstrainedCodomain{
            y: self.get_y(o),
            constraints: self.get_constraints(o),
        }
    }
}

impl <Out:Outcome> Single<Out> for ConstrainedCodomain<Out>
{
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}

impl <Out:Outcome> Constrained<Out,ConsType> for ConstrainedCodomain<Out>
{
    fn get_criteria(&self) -> &Box<[Criteria<Out>]> {
        &self.c_criteria
    }
}

/// A [`Single`], [`Constrained`], and [`Fidelity`] [`Codomain`].
pub struct FidelConstCodomain<Out:Outcome>
{
    pub y_criteria : Criteria<Out>,
    pub f_criteria : Criteria<Out>,
    pub c_criteria : Box<[Criteria<Out>]>,
}
/// A element ([`TypeCodom`](Codomain::TypeDom)) from [`ConstrainedCodomain`].
pub struct ElemFidelConstCodomain{
    pub y : f64,
    pub fidelity : f64,
    pub constraints : Box<[f64]>,
}

impl <Out:Outcome> Codomain<Out> for FidelConstCodomain<Out>{
    type TypeCodom=ElemFidelConstCodomain;

    fn get_elem(&self, o:&Out)-> Self::TypeCodom {
        ElemFidelConstCodomain{
            y: self.get_y(o),
            fidelity: self.get_fidelity(o),
            constraints: self.get_constraints(o),
        }
    }
}
impl <Out:Outcome> Single<Out> for FidelConstCodomain<Out>{
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}
impl <Out:Outcome> Fidelity<Out> for FidelConstCodomain<Out>{
    fn get_criteria(&self) -> Criteria<Out> {
        self.f_criteria
    }
}
impl <Out:Outcome> Constrained<Out,ConsType> for FidelConstCodomain<Out>{
    fn get_criteria(&self) -> &Box<[Criteria<Out>]> {
        &self.c_criteria
    }
}

pub type ConstFidelCodomain<Out> = FidelConstCodomain<Out>;


// MULTI OBJECTIVE CODOMAINS

/// A [`Multi`] objective [`Codomain`].
pub struct MultiCodomain<Out:Outcome>
{
    pub y_criteria : Box<[Criteria<Out>]>,
}

impl <Out:Outcome> Codomain<Out> for MultiCodomain<Out>{
    type TypeCodom = Box<[f64]>;

    fn get_elem(&self, o:&Out)-> Self::TypeCodom {
        self.get_y(o)
    }
}

impl <Out:Outcome> Multi<Out> for MultiCodomain<Out>{
    fn get_criteria(&self) -> &Box<[Criteria<Out>]> {
        &self.y_criteria
    }
}

/// A [`Multi`] objective and [`Fidelity`] [`Codomain`].
pub struct FidelMultiCodomain<Out:Outcome>
{
    pub y_criteria : Box<[Criteria<Out>]>,
    pub f_criteria : Criteria<Out>,
}
/// A element ([`TypeCodom`](Codomain::TypeDom)) from [`ConstrainedCodomain`].
pub struct ElemFidelMultiCodomain{
    pub y : Box<[f64]>,
    pub fidelity : f64,
}

impl <Out:Outcome> Codomain<Out> for FidelMultiCodomain<Out>{
    type TypeCodom=ElemFidelMultiCodomain;

    fn get_elem(&self, o:&Out)-> Self::TypeCodom {
        ElemFidelMultiCodomain{
            y: self.get_y(o),
            fidelity: self.get_fidelity(o),
        }
    }
}
impl <Out:Outcome> Multi<Out> for FidelMultiCodomain<Out>{
    fn get_criteria(&self) -> &Box<[Criteria<Out>]> {
        &self.y_criteria
    }
}
impl <Out:Outcome> Fidelity<Out> for FidelMultiCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.f_criteria
    }
}

/// A [`Multi`] objective and [`Constrained`] [`Codomain`].
pub struct ConstMultiCodomain<Out:Outcome>
{
    pub y_criteria : Box<[Criteria<Out>]>,
    pub c_criteria : Box<[Criteria<Out>]>,
}
/// A element ([`TypeCodom`](Codomain::TypeDom)) from [`ConstrainedCodomain`].
pub struct ElemConstMultiCodomain{
    pub y : Box<[f64]>,
    pub constraint : Box<[f64]>,
}

impl <Out:Outcome> Codomain<Out> for ConstMultiCodomain<Out>{
    type TypeCodom=ElemConstMultiCodomain;

    fn get_elem(&self, o:&Out)-> Self::TypeCodom {
        ElemConstMultiCodomain{
            y: self.get_y(o),
            constraint: self.get_constraints(o),
        }
    }
}

impl <Out:Outcome> Multi<Out> for ConstMultiCodomain<Out>{
    fn get_criteria(&self) -> &Box<[Criteria<Out>]> {
        &self.y_criteria
    }
}
impl <Out:Outcome> Constrained<Out,ConsType> for ConstMultiCodomain<Out>{
    fn get_criteria(&self) -> &Box<[Criteria<Out>]> {
        &self.c_criteria
    }
}

/// A [`Multi`] objective, [`Constrained`], and [`Fidelity`] [`Codomain`].
pub struct FidelConstMultiCodomain<Out:Outcome>
{
    pub y_criteria : Box<[Criteria<Out>]>,
    pub f_criteria : Criteria<Out>,
    pub c_criteria : Box<[Criteria<Out>]>,
}
/// A element ([`TypeCodom`](Codomain::TypeDom)) from [`ConstrainedCodomain`].
pub struct ElemFidelConstMultiCodomain{
    pub y : Box<[f64]>,
    pub fidelity : f64,
    pub constraints : Box<[f64]>,
}

impl <Out:Outcome> Codomain<Out> for FidelConstMultiCodomain<Out>{
    type TypeCodom=ElemFidelConstMultiCodomain;

    fn get_elem(&self, o:&Out)-> Self::TypeCodom {
        ElemFidelConstMultiCodomain{
            y: self.get_y(o),
            fidelity: self.get_fidelity(o),
            constraints: self.get_constraints(o),
        }
    }
}
impl <Out:Outcome> Multi<Out> for FidelConstMultiCodomain<Out>{
    fn get_criteria(&self) -> &Box<[Criteria<Out>]> {
        &self.y_criteria
    }
}

impl <Out:Outcome> Fidelity<Out> for FidelConstMultiCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.f_criteria
    }
}

impl <Out:Outcome> Constrained<Out,ConsType> for FidelConstMultiCodomain<Out>{
    fn get_criteria(&self) -> &Box<[Criteria<Out>]> {
        &self.c_criteria
    }
}

pub type ConstFidelMultiCodomain<Out> = FidelConstMultiCodomain<Out>;