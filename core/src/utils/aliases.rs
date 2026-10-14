//! Type aliases for convenience in accessing associated types.

use crate::{Codomain, Domain, FuncWrapper, HasId, HasInfo, HasX, HasY, Linked, Outcome, Problem, SolutionShape, solution::shape::IntoComputedShape};


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
/// The [`TypeDom`](Domain::TypeDom) of the `Obj` domain linked to `Self` through the [`Linked`] trait.
pub type LinkTyObj<T> = <<T as Linked>::Obj as Domain>::Element;
/// The [`TypeDom`](Domain::TypeDom) of the `Opt` domain linked to `Self` through the [`Linked`] trait.
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
/// Alias for the solution metadata type of a [`HasSolInfo`] object.
pub type GetSInfo<T> = <T as HasInfo>::Info;
/// Alias for the ID type of a [`HasId`] object.
pub type GetId<T> = <T as HasId>::Id;
/// Alias for the Y type of a [`HasY`] object.
pub type GetY<T> = <T as HasY>::Y;

// EXECUTION
// OBJECTIVE FUNCTION
pub type GetFuncSol<Func,Dom> = <Func as FuncWrapper>::Sol<Dom>;
pub type Func<Pb> = <Pb as Problem>::Func;
pub type GetFuncCtx<Pb> = <Func<Pb> as FuncWrapper>::Context;
pub type GetFuncOutShape<Pb> = <Func<Pb> as FuncWrapper>::OutShape;
pub type Out<Pb> = <Func<Pb> as FuncWrapper>::Out;
// DOMAIN
pub type Scp<Pb> = <Pb as Problem>::Scp;
pub type Obj<Pb> = LinkObj<Scp<Pb>>;
pub type TObj<Pb> = LinkTyObj<Scp<Pb>>;
pub type Opt<Pb> = LinkOpt<Scp<Pb>>;
pub type TOpt<Pb> = LinkTyOpt<Scp<Pb>>;
pub type TrueOpt<Pb> = LinkTrueOpt<Scp<Pb>>;
// CODOMAIN
pub type Cod<Pb> = <Out<Pb> as Outcome>::Cod;
pub type CodElem<Pb> = <Cod<Pb> as Codomain<Out<Pb>>>::TypeCodom;
// SOLUTION
pub type Id<Pb> = <Pb as Problem>::Id;
pub type SObj<Pb> = <Func<Pb> as FuncWrapper>::Sol<Obj<Pb>>;
pub type XObj<Pb> = <SObj<Pb> as HasX>::X;
pub type SOpt<Pb> = <Func<Pb> as FuncWrapper>::Sol<Opt<Pb>>;
pub type XOpt<Pb> = <SOpt<Pb> as HasX>::X;
pub type Shape<Pb> = <Pb as Problem>::Shape;
pub type Accum<Pb> = <Cod<Pb> as Codomain<Out<Pb>>>::Acc<CompShape<Pb>>;
pub type CompShape<Pb> = <Shape<Pb> as IntoComputedShape>::ComputedShape<Out<Pb>>;
pub type Info<Pb> = <Pb as Problem>::Info;