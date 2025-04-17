use crate::core::domain::Domain;
use crate::core::domain::bounded::Bounded;
use crate::core::domain::unit::Unit;
use crate::core::domain::bool::Bool;
use crate::core::domain::cat::Cat;
use crate::core::domain::errors_domain::DomainError;
use crate::core::domain::onto::Onto;

use num::cast::AsPrimitive;
use num::{Float, Num, NumCast};
use rand::distr::uniform::SampleUniform;
use rand::prelude::ThreadRng;
use std::fmt::{self, Debug, Display};
use std::ops::RangeInclusive;


// -_-_-_-_-_-_-_-
// Grouped domains

pub enum BaseDom<'a,const N : usize,T,U>
where
    T: Num
    + NumCast
    + PartialEq
    + PartialOrd
    + Clone
    + SampleUniform
    + AsPrimitive<f64>
    + Display
    + Debug,
    U : Num
    + NumCast
    + Float
    + PartialEq
    + PartialOrd
    + Clone
    + SampleUniform
    + AsPrimitive<f64>
    + Display
    + Debug,
{
    Bounded(Bounded<T>),
    Bool(Bool),
    Cat(Cat<'a,N>),
    Unit(Unit<U>)
}

#[derive(Copy,Clone,PartialEq)]
pub enum BaseTypeDom<'a, const N : usize,T, U>
where
    T: Num
    + NumCast
    + PartialEq
    + PartialOrd
    + Clone
    + SampleUniform
    + AsPrimitive<f64>
    + Display
    + Debug,
    U : Num
    + NumCast
    + Float
    + PartialEq
    + PartialOrd
    + Clone
    + SampleUniform
    + AsPrimitive<f64>
    + Display
    + Debug,
{
    Bounded(<Bounded<T> as Domain>::TypeDom),
    Bool(<Bool as Domain>::TypeDom),
    Cat(<Cat<'a,N> as Domain>::TypeDom),
    Unit(<Unit<U> as Domain>::TypeDom),
}

// impl<'a, const N:usize, T> Display for BaseTypeDom<'a, N, T>
// where
//     T: Num + NumCast,
//     T: PartialEq + PartialOrd,
//     T: SampleUniform,
//     T: Clone + Copy,
//     T: Display + Debug,
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {

//         match self {
//             Self::Bounded(d) => std::fmt::Display::fmt(&d, f),
//             Self::Bool(d) => std::fmt::Display::fmt(&d, f),
//             Self::Cat(d) => std::fmt::Display::fmt(&d, f),
//         }
//     }
// }
// impl<'a, const N:usize,T> Debug for BaseTypeDom<'a, N,T>
// where
//     T: Num + NumCast,
//     T: PartialEq + PartialOrd,
//     T: SampleUniform,
//     T: Clone + Copy,
//     T: Display + Debug,
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {

//         match self {
//             Self::Bounded(d) => std::fmt::Debug::fmt(&d, f),
//             Self::Bool(d) => std::fmt::Debug::fmt(&d, f),
//             Self::Cat(d) => std::fmt::Debug::fmt(&d, f),
//         }
//     }
// }

// impl <'a, const N : usize,T> Domain for BaseDom<'a, N,T>
// where
//     T: Num + NumCast,
//     T: PartialEq + PartialOrd,
//     T: SampleUniform,
//     T: Clone + Copy,
//     T: Display + Debug,
// {
//     type TypeDom = BaseTypeDom<'a,N,T>;
//     fn default_sampler(&self) -> fn(&Self, &mut ThreadRng) -> Self::TypeDom {
//         |d,rng| d.default_sampler()(d,rng)
//     }

//     fn is_in(&self, item: &Self::TypeDom) -> bool {
//         match self {
//             Self::Bounded(d) => match item{
//                 Self::TypeDom::Bounded(i) => d.is_in(i),
//                 _ => false,
//             },
//             Self::Bool(d) => match item{
//                 Self::TypeDom::Bool(i) => d.is_in(i),
//                 _ => false,
//             },
//             Self::Cat(d) => match item{
//                 Self::TypeDom::Cat(i) => d.is_in(i),
//                 _ => false,
//             },
//         }
//     }
// }

// impl<'a, const N: usize, T, Out> Onto<Bounded<Out>> for BaseDom<'a, N, T>
// where
//     Out: Num
//     + NumCast
//     + PartialEq
//     + PartialOrd
//     + Clone
//     + SampleUniform
//     + AsPrimitive<f64>
//     + Display
//     + Debug,
//     T: Num + NumCast,
//     T: PartialEq + PartialOrd,
//     T: SampleUniform,
//     T: Clone + Copy,
//     T: Display + Debug,
// {
//     /// [`Onto`] function between a [`Cat`] and a [`Bounded`] [`Domain`].
//     ///
//     /// Considering $l_{out}$ and $u_{out}$ the lower and upper bounds of
//     /// the the output [`Bounded`] [`Domain`], and $i$ the index of the item within `values` of [`Cat`].
//     /// The mapping is given by :
//     ///
//     /// $$ i/|\texttt(values)| \time (u_{out}-l_{out}) + l_{out} $$
//     ///
//     /// # Parameters
//     ///
//     /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
//     /// * `target` : `&`[`Bounded`]`<Out>` - A borrowed targetted [`Domain`].
//     ///
//     /// # Errors
//     ///
//     /// * Returns a [`DomainError`]
//     ///     * if input `item` to be mapped is not into [`Self`] domain.
//     ///     * if resulting mapped `item` is not into the `target` domain.
//     ///
//     fn onto(
//         &self,
//         item: &<BaseDom<'a, N, T> as Domain>::TypeDom,
//         target: &Bounded<Out>,
//     ) -> Result<Out, DomainError> {
//         match self {
//             Self::Bounded(d) => match item{
//                 Self::TypeDom::Bounded(i) => d.onto(i,target),
//                 _ => false,
//             },
//             Self::Bool(d) => match item{
//                 Self::TypeDom::Bool(i) => d.is_in(i),
//                 _ => false,
//             },
//             Self::Cat(d) => match item{
//                 Self::TypeDom::Cat(i) => d.is_in(i),
//                 _ => false,
//             },
//         }
//     }
// }