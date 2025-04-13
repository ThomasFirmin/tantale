use crate::core::domain::{Bool, Bounded, Cat, Domain, DomainBounded};
use crate::core::errors::DomainError;

use num::{Num, NumCast};
use rand::distr::uniform::SampleUniform;
use std::fmt::{Debug, Display};

use num::cast::AsPrimitive;

pub trait Onto<Out: Domain>: Domain {
    /// [`Onto`] is a surjective function to map a point from an input [`Domain`] to an output [`Domain`].
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]``>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Domain`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, _item: &Self::TypeDom, _target: &Out) -> Result<Out::TypeDom, DomainError> {
        Err(DomainError {
            code: 101,
            msg: format!(
                "Consider implement Onto<{}> for {}",
                std::any::type_name::<Self>(),
                std::any::type_name::<Out>()
            ),
        })
    }
}

impl<In, Out> Onto<Bounded<Out>> for Bounded<In>
where
    In: Num
        + NumCast
        + PartialEq
        + PartialOrd
        + Clone
        + SampleUniform
        + AsPrimitive<f64>
        + Display
        + Debug,
    Out: Num
        + NumCast
        + PartialEq
        + PartialOrd
        + Clone
        + SampleUniform
        + AsPrimitive<f64>
        + Display
        + Debug,
    f64: AsPrimitive<Out>,
{
    /// [`Onto`] function between [`Bounded`] [`Domain`].
    ///
    /// Considering $l_{in}$, $u_{in}$, $l_{out}$ and $u_{out}$ the lower and upper bounds of
    /// the input [`Bounded`] [`Domain`] and the output [`Bounded`] [`Domain`], and $x$ the item to be mapped,
    /// the mapping is given by
    ///
    /// $$ \frac{x-l_{in}}{u_{in}-l_{in}} \times (u_{out}-l_{out}) + l_{out}$$
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Bounded`]`<Out>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &In, target: &Bounded<Out>) -> Result<Out, DomainError> {
        if self.is_in(item) {
            let a: f64 = (*item - self.lower()).as_();
            let b: f64 = self.width().as_();
            let c: f64 = target.width().as_();
            let mapped: Out = (a / b * c).as_() + target.lower();

            if target.is_in(&mapped) {
                Ok(mapped)
            } else {
                Err(DomainError {
                    code: 101,
                    msg: format!("{} -> {} not in {}", item, mapped, target),
                })
            }
        } else {
            Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            })
        }
    }
}

impl<In> Onto<Bool> for Bounded<In>
where
    In: Num
        + NumCast
        + PartialEq
        + PartialOrd
        + Clone
        + SampleUniform
        + AsPrimitive<f64>
        + Display
        + Debug,
    f64: AsPrimitive<In>,
{
    /// [`Onto`] function between [`Bounded`] [`Domain`].
    ///
    /// Considering $l$ and $u$ the lower and upper bounds of
    /// the input [`Bounded`] [`Domain`], and $x$ the `item`, returns `true` if $x>\frac{u-l}{2}$
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Bool`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &In, _target: &Bool) -> Result<<Bool as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            Ok(*item > self.mid())
        } else {
            return Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            });
        }
    }
}

impl<'a, In, const N: usize> Onto<Cat<'a, N>> for Bounded<In>
where
    In: Num
        + NumCast
        + PartialEq
        + PartialOrd
        + Clone
        + SampleUniform
        + AsPrimitive<f64>
        + Display
        + Debug,
    f64: AsPrimitive<In>,
{
    /// [`Onto`] function between [`Bounded`] [`Domain`].
    ///
    /// Considering $l_{in}$, $u_{in}$ the lower and upper bounds of
    /// the input [`Bounded`] [`Domain`] and $\ell_{en}$ the length of `values` of the [`Cat`] [`Domain`]. The variable $x$ is the item to be mapped.
    /// The mapping is given by mapping item to an index of `values` in [`Cat`]:
    ///
    /// $$ \texttt{Floor}\Biggl(\frac{x-l_{in}}{u_{in}-l_{in}} \times (\ell_{en}-1)\Biggl) $$
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Cat`]<'a, N> - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &In,
        target: &Cat<'a, N>,
    ) -> Result<<Cat<'a, N> as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let a: f64 = (*item - self.lower()).as_();
            let b: f64 = self.width().as_();
            let c: f64 = (target.values().len() - 1).as_();
            let mapped = target.values()[(a / b * c) as usize];

            if target.is_in(&mapped) {
                Ok(mapped)
            } else {
                Err(DomainError {
                    code: 101,
                    msg: format!("{} -> {} not in {}", item, mapped, target),
                })
            }
        } else {
            Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            })
        }
    }
}

impl<Out> Onto<Bounded<Out>> for Bool
where
    Out: Num
        + NumCast
        + PartialEq
        + PartialOrd
        + Clone
        + SampleUniform
        + AsPrimitive<f64>
        + Display
        + Debug,
    f64: AsPrimitive<Out>,
{
    /// [`Onto`] function between a [`Bool`] and a [`Bounded`] [`Domain`].
    ///
    /// Considering $l_{out}$ and $u_{out}$ the lower and upper bounds of
    /// the the output [`Bounded`] [`Domain`], and $x$ the item to be mapped.
    /// If `item` is `true` returns $u_{out}$, otherwise returns $l_{out}$ .
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Bounded`]`<Out>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &<Bool as Domain>::TypeDom,
        target: &Bounded<Out>,
    ) -> Result<Out, DomainError> {
        if self.is_in(item) {
            let mapped = if *item {
                target.upper()
            } else {
                target.lower()
            };
            if target.is_in(&mapped) {
                Ok(mapped)
            } else {
                Err(DomainError {
                    code: 101,
                    msg: format!("{} -> {} not in {}", item, mapped, target),
                })
            }
        } else {
            Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            })
        }
    }
}

impl Onto<Bool> for Bool {
    /// [`Onto`] function between a [`Bool`] and another [`Bool`] [`Domain`].
    ///
    /// Useless, return `true` if `*item` is `true`, `false` otherwise.
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Bool`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &<Bool as Domain>::TypeDom, _target: &Bool) -> Result<bool, DomainError> {
        Ok(*item)
    }
}

impl<'a, const N: usize, Out> Onto<Bounded<Out>> for Cat<'a, N>
where
    Out: Num
        + NumCast
        + PartialEq
        + PartialOrd
        + Clone
        + SampleUniform
        + AsPrimitive<f64>
        + Display
        + Debug,
    f64: AsPrimitive<Out>,
{
    /// [`Onto`] function between a [`Cat`] and a [`Bounded`] [`Domain`].
    ///
    /// Considering $l_{out}$ and $u_{out}$ the lower and upper bounds of
    /// the the output [`Bounded`] [`Domain`], and $i$ the index of the item within `values` of [`Cat`].
    /// The mapping is given by :
    ///
    /// $$ i/|\texttt(values)| \time (u_{out}-l_{out}) + l_{out} $$
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]`>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Bounded`]`<Out>` - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(
        &self,
        item: &<Cat<'a, N> as Domain>::TypeDom,
        target: &Bounded<Out>,
    ) -> Result<Out, DomainError> {
        let idx = self.values().iter().position(|n| n == item);

        match idx {
            Some(i) => {
                let a: f64 = (i).as_();
                let b: f64 = (self.values().len() - 1).as_();
                let c: f64 = target.width().as_();
                let mapped: Out = (a / b * c).as_() + target.lower();

                if target.is_in(&mapped) {
                    return Ok(mapped);
                } else {
                    return Err(DomainError {
                        code: 101,
                        msg: format!("{} -> {} not in {}", item, mapped, target),
                    });
                }
            }
            None => {
                return Err(DomainError {
                    code: 103,
                    msg: format!("{} not in {}", item, self),
                })
            }
        }
    }
}
