//! Type aliases for convenience in accessing associated types.

use crate::{
    Codomain, Domain, FuncWrapper, HasId, HasInfo, HasX, HasY, Linked, Outcome, Problem, Solution,
    SolutionShape, solution::shape::IntoComputedShape,
};

/// Type alias for extracting the value type from a domain.
///
/// [`Elem<T>`] is a convenience type alias that maps a domain type `T` to its associated
/// [`Domain::Element`] type. This simplifies type signatures and makes code more readable.
///
/// # See Also
///
/// - [`Domain::Element`] - The associated type being aliased
/// - [`Domain`] - The trait defining domains
pub type Elem<T> = <T as Domain>::Element;

// LINKED
/// The type of the `Obj` domain linked to `Self` through the [`Linked`] trait.
pub type LinkObj<T> = <T as Linked>::Obj;
/// The type of the `Opt` domain linked to `Self` through the [`Linked`] trait.
pub type LinkOpt<T> = <T as Linked>::Opt;
/// The [`Element`](Domain::Element) of the `Obj` domain linked to `Self` through the [`Linked`] trait.
pub type LinkTyObj<T> = <<T as Linked>::Obj as Domain>::Element;
/// The [`Element`](Domain::Element) of the `Opt` domain linked to `Self` through the [`Linked`] trait.
pub type LinkTyOpt<T> = <<T as Linked>::Opt as Domain>::Element;
/// The type of the `TrueOpt` domain linked to `Self` through the [`Linked`] trait.
pub type LinkTrueOpt<T> = <T as Linked>::TrueOpt;

// CODOMAIN

/// [`TypeCodom`](Codomain::TypeCodom) of a [`Codomain`].
pub type TypeCodom<Out> = <<Out as Outcome>::Cod as Codomain<Out>>::TypeCodom;

// SolutionShape

/// The type of the objective solution associated with a [`SolutionShape`].
pub type GetSObj<S> = <S as SolutionShape>::ObjSol;
/// The type of the optimizer solution associated with a [`SolutionShape`].
pub type GetSOpt<S> = <S as SolutionShape>::OptSol;

/// Alias for the raw type of a [`HasX`] object.
pub type GetRaw<T> = <T as HasX>::X;
/// Alias for the solution metadata type of a [`HasInfo`] object.
pub type GetSInfo<T> = <T as HasInfo>::Info;
/// Alias for the ID type of a [`HasId`] object.
pub type GetId<T> = <T as HasId>::Id;
/// Alias for the Y type of a [`HasY`] object.
pub type GetY<T> = <T as HasY>::Y;

// EXECUTION
// OBJECTIVE FUNCTION

/// Type aliases for convenience in accessing associated [`MetaData`] types of a [`Solution`].
pub type MetaData<Sol> = <Sol as Solution>::MetaData;
/// Type aliases for convenience in accessing associated [`Solution`] of a [`FuncWrapper`].
pub type GetFuncSol<Func, Dom> = <Func as FuncWrapper>::Sol<Dom>;
/// Type aliases for convenience in accessing associated [`FuncWrapper`] of a [`Problem`].
pub type Func<Pb> = <Pb as Problem>::Func;
/// Type aliases for convenience in accessing associated [`Context`](FuncWrapper::Context) of a [`FuncWrapper`].
pub type GetFuncCtx<Pb> = <Func<Pb> as FuncWrapper>::Context;
/// Type aliases for convenience in accessing associated [`OutShape`](FuncWrapper::OutShape) of a [`FuncWrapper`].
pub type GetFuncOutShape<Pb> = <Func<Pb> as FuncWrapper>::OutShape;
/// Type aliases for convenience in accessing associated [`Out`](FuncWrapper::Out) of a [`FuncWrapper`].
pub type Out<Pb> = <Func<Pb> as FuncWrapper>::Out;

// DOMAIN

/// Type aliases for convenience in accessing associated [`Searchspace`](Problem::Scp) of a [`Problem`].
pub type Scp<Pb> = <Pb as Problem>::Scp;
/// Type aliases for convenience in accessing associated the objective-facing [`Domain`] of a [`Problem`].
pub type Obj<Pb> = LinkObj<Scp<Pb>>;
/// Type aliases for convenience in accessing associated the objective-facing [`Element`](Domain::Element) of a [`Problem`].
pub type TObj<Pb> = LinkTyObj<Scp<Pb>>;
/// Type aliases for convenience in accessing associated the optimizer-facing [`Domain`] of a [`Problem`].
pub type Opt<Pb> = LinkOpt<Scp<Pb>>;
/// Type aliases for convenience in accessing associated the optimizer-facing [`Element`](Domain::Element) of a [`Problem`].
pub type TOpt<Pb> = LinkTyOpt<Scp<Pb>>;
/// Type aliases for convenience in accessing associated the true optimizer-facing [`Domain`] of a [`Problem`].
/// This is different from [`Opt<Pb>`] when the optimizer-facing domain is a [`NoDomain`](crate::NoDomain), [`Opt<Pb>`] gives [`Obj<Pb>`] while
/// [`TrueOpt<Pb>`] returns [`NoDomain`](crate::NoDomain).
pub type TrueOpt<Pb> = LinkTrueOpt<Scp<Pb>>;

// CODOMAIN

/// Type aliases for convenience in accessing associated [`Codomain`] of a [`Problem`].
pub type Cod<Pb> = <Out<Pb> as Outcome>::Cod;
/// Type aliases for convenience in accessing associated [`Codomain::TypeCodom`] of an [`Problem`].
pub type CodElem<Pb> = <Cod<Pb> as Codomain<Out<Pb>>>::TypeCodom;

// SOLUTION

/// Type aliases for convenience in accessing associated [`SolId`](crate::SolId) of a [`Problem`].
pub type Id<Pb> = <Pb as Problem>::Id;
/// Type aliases for convenience in accessing associated objective-facing [`Solution`] of a [`Problem`].
pub type SObj<Pb> = <Func<Pb> as FuncWrapper>::Sol<Obj<Pb>>;
/// Type aliases for convenience in accessing associated objective-facing [`X`](HasX::X) (raw solution) of a [`Problem`].
pub type XObj<Pb> = <SObj<Pb> as HasX>::X;
/// Type aliases for convenience in accessing associated optimizer-facing [`Solution`] of a [`Problem`].
pub type SOpt<Pb> = <Func<Pb> as FuncWrapper>::Sol<Opt<Pb>>;
/// Type aliases for convenience in accessing associated optimizer-facing [`X`](HasX::X) (raw solution) of a [`Problem`].
pub type XOpt<Pb> = <SOpt<Pb> as HasX>::X;
/// Type aliases for convenience in accessing associated [`SolutionShape`] of a [`Problem`].
pub type Shape<Pb> = <Pb as Problem>::Shape;
/// Type aliases for convenience in accessing associated [`ComputedShape`](crate::IntoComputedShape::ComputedShape) of a [`Problem`].
pub type CompShape<Pb> = <Shape<Pb> as IntoComputedShape>::ComputedShape<Out<Pb>>;
/// Type aliases for convenience in accessing associated [`Accumulator`](Codomain::Acc) of a [`Problem`].
pub type Accum<Pb> = <Cod<Pb> as Codomain<Out<Pb>>>::Acc<CompShape<Pb>>;
/// Type aliases for convenience in accessing associated [`Info`](crate::SolInfo) of a [`Problem`].
pub type Info<Pb> = <Pb as Problem>::Info;
