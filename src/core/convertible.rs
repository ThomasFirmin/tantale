
use crate::core::domain::{Bool, Bounded, Cat, Domain, NumericallyBounded, Real, Int, Nat};
use crate::core::errors::DomainError;

use num::{Num, NumCast};
use rand::distr::uniform::SampleUniform;
use std::fmt::Display;

use num::cast::AsPrimitive;

pub trait OnTo<Out:Domain> {
    /// [`OnTo`] is a surjective function to map a point from an input [`Domain`] to an output [`Domain`]
    ///
    /// # Parameters
    ///
    /// * `item` : <Self as Domain>::TypeDom - The point from the [`Self`] domain to map to the `target` [`Real`] domain.
    /// * `target` : &Real - Reference to the targeted domain.
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self,_item:&<Self as Domain>::TypeDom,_target: &Out)->Result<Out::TypeDom, DomainError>
    where
        Self: Domain
    {
        Err(DomainError {
            code: 101,
            msg: format!(
                "Consider implement OnTo<{}> for {}",
                std::any::type_name::<Self>(),std::any::type_name::<Out>()
            ),
        })
    }
}

impl <In,Out> OnTo<Bounded<Out>> for Bounded<In>
where
    In: Num + NumCast + PartialOrd + Clone + SampleUniform+AsPrimitive<f64> + Display,
    Out: Num + NumCast + PartialOrd + Clone + SampleUniform+AsPrimitive<f64> + Display,
    f64: AsPrimitive<Out>,
{
    fn onto(&self,item:&<Self as Domain>::TypeDom,target: &Bounded<Out>)->Result<<Bounded<Out> as Domain>::TypeDom, DomainError> 
    {
        
        if self.is_in(item) {
            let a:f64 = (*item - self.lower()).as_();
            let b:f64 = self.width().as_();
            let c:f64 = target.width().as_();
            let mapped:<Bounded<Out> as Domain>::TypeDom = (a / b * c).as_() + target.lower();
    
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

// impl<In> Convertible for Bounded<In> {
//     /// See [`Convertible`] for more infos.
//     /// Considering $m$ as the middle point of the [`Real`] domain, the mapping is given by:
//     /// $$\texttt{item} > m$$
//     fn to_bool(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         _target: &Bool,
//     ) -> Result<<Bool as Domain>::TypeDom, DomainError> 
//     where
//         Self: Domain + NumericallyBounded + Display,
//         <Self as Domain>::TypeDom: Num + NumCast + PartialOrd +Display,
//         In: Num + NumCast ,
//         {
//         if self.is_in(item) {
//             Ok(*item > self.mid())
//         } else {
//             return Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             });
//         }
//     }
// }



// impl Convertible for Nat {
//     /// See [`Convertible`] for more infos.
//     /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Real`] domain, the mapping is given by:
//     /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
//     fn to_real(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Real,
//     ) -> Result<<Real as Domain>::TypeDom, DomainError> {
//         if self.is_in(item) {
//             let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom)
//                 / (self.range() as <Real as Domain>::TypeDom);
//             let mapped = normalized * target.range() + target.lower();

//             if target.is_in(&mapped) {
//                 Ok(mapped)
//             } else {
//                 Err(DomainError {
//                     code: 101,
//                     msg: format!("{} -> {} not in {}", item, mapped, target),
//                 })
//             }
//         } else {
//             Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             })
//         }
//     }
//     /// See [`Convertible`] for more infos.
//     /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Nat`] domain, the mapping is given by:
//     /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
//     fn to_nat(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Nat,
//     ) -> Result<<Nat as Domain>::TypeDom, DomainError> {
//         if self.is_in(item) {
//             let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom)
//                 / (self.range() as <Real as Domain>::TypeDom);
//             let mapped = (normalized * (target.range() as <Real as Domain>::TypeDom))
//                 as <Nat as Domain>::TypeDom
//                 + target.lower();

//             if target.is_in(&mapped) {
//                 Ok(mapped)
//             } else {
//                 Err(DomainError {
//                     code: 101,
//                     msg: format!("{} -> {} not in {}", item, mapped, target),
//                 })
//             }
//         } else {
//             Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             })
//         }
//     }
//     /// See [`Convertible`] for more infos.
//     /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Int`] domain, the mapping is given by:
//     /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
//     fn to_int(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Int,
//     ) -> Result<<Int as Domain>::TypeDom, DomainError> {
//         if self.is_in(item) {
//             let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom)
//                 / (self.range() as <Real as Domain>::TypeDom);
//             let mapped = (normalized * (target.range() as <Real as Domain>::TypeDom))
//                 as <Int as Domain>::TypeDom
//                 + target.lower();

//             if target.is_in(&mapped) {
//                 Ok(mapped)
//             } else {
//                 Err(DomainError {
//                     code: 101,
//                     msg: format!("{} -> {} not in {}", item, mapped, target),
//                 })
//             }
//         } else {
//             Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             })
//         }
//     }
//     /// See [`Convertible`] for more infos.
//     /// Considering $m$ as the middle point of the [`Real`] domain, the mapping is given by:
//     /// $$\texttt{item} > m$$
//     fn to_bool(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         _target: &Bool,
//     ) -> Result<<Bool as Domain>::TypeDom, DomainError> {
//         if self.is_in(item) {
//             Ok(*item > self.mid())
//         } else {
//             return Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             });
//         }
//     }

//     /// See [`Converter`] for more infos.
//     /// Maps a float to an element of `values` from the `target` [`Cat`].
//     /// Convert a float $x$ into an index. Considering $u$ and $l$ as the lower and upper bounds of the [`Nat`], we have:
//     /// $$ \texttt{index} = \frac{x - l}{u-l}*\texttt{values}.len()} $$
//     ///
//     /// # Notes
//     ///
//     /// The normalization of the previous equation is casted into [`f64`], then recasted into a [`usize`] for the index.
//     ///
//     fn to_cat<'a, const N: usize>(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Cat<'a, N>,
//     ) -> Result<<Cat<'a, N> as Domain>::TypeDom, DomainError> {
//         if self.is_in(item) {
//             let idx: usize;
//             if *item < self.upper() {
//                 idx = (((*item - self.lower()) as f64) / (self.range() as f64)
//                     * (target.values().len() as f64)) as usize;
//             } else {
//                 idx = target.values().len() - 1;
//             }
//             Ok(target.values()[idx])
//         } else {
//             Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             })
//         }
//     }
// }

// impl Convertible for Int {
//     /// See [`Convertible`] for more infos.
//     /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Real`] domain, the mapping is given by:
//     /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
//     fn to_real(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Real,
//     ) -> Result<<Real as Domain>::TypeDom, DomainError> {
//         if self.is_in(item) {
//             let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom)
//                 / (self.range() as <Real as Domain>::TypeDom);
//             let mapped = normalized * target.range() + target.lower();

//             if target.is_in(&mapped) {
//                 Ok(mapped)
//             } else {
//                 Err(DomainError {
//                     code: 101,
//                     msg: format!("{} -> {} not in {}", item, mapped, target),
//                 })
//             }
//         } else {
//             Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             })
//         }
//     }
//     /// See [`Convertible`] for more infos.
//     /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Nat`] domain, the mapping is given by:
//     /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
//     fn to_nat(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Nat,
//     ) -> Result<<Nat as Domain>::TypeDom, DomainError> {
//         if self.is_in(item) {
//             let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom)
//                 / (self.range() as <Real as Domain>::TypeDom);
//             let mapped = (normalized * (target.range() as <Real as Domain>::TypeDom))
//                 as <Nat as Domain>::TypeDom
//                 + target.lower();

//             if target.is_in(&mapped) {
//                 Ok(mapped)
//             } else {
//                 Err(DomainError {
//                     code: 101,
//                     msg: format!("{} -> {} not in {}", item, mapped, target),
//                 })
//             }
//         } else {
//             Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             })
//         }
//     }
//     /// See [`Convertible`] for more infos.
//     /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Int`] domain, the mapping is given by:
//     /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
//     fn to_int(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Int,
//     ) -> Result<<Int as Domain>::TypeDom, DomainError> {
//         if self.is_in(item) {
//             let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom)
//                 / (self.range() as <Real as Domain>::TypeDom);
//             let mapped = (normalized * (target.range() as <Real as Domain>::TypeDom))
//                 as <Int as Domain>::TypeDom
//                 + target.lower();

//             if target.is_in(&mapped) {
//                 Ok(mapped)
//             } else {
//                 Err(DomainError {
//                     code: 101,
//                     msg: format!("{} -> {} not in {}", item, mapped, target),
//                 })
//             }
//         } else {
//             Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             })
//         }
//     }
//     /// See [`Convertible`] for more infos.
//     /// Considering $m$ as the middle point of the [`Real`] domain, the mapping is given by:
//     /// $$\texttt{item} > m$$
//     fn to_bool(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         _target: &Bool,
//     ) -> Result<<Bool as Domain>::TypeDom, DomainError> {
//         if self.is_in(item) {
//             Ok(*item > self.mid())
//         } else {
//             return Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             });
//         }
//     }

//     /// See [`Converter`] for more infos.
//     /// Maps a float to an element of `values` from the `target` [`Cat`].
//     /// Convert a float $x$ into an index. Considering $u$ and $l$ as the lower and upper bounds of the [`Nat`], we have:
//     /// $$ \texttt{index} = \frac{x - l}{u-l}*\texttt{values}.len()} $$
//     ///
//     /// # Notes
//     ///
//     /// The normalization of the previous equation is casted into [`f64`], then recasted into a [`usize`] for the index.
//     ///
//     fn to_cat<'a, const N: usize>(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Cat<'a, N>,
//     ) -> Result<<Cat<'a, N> as Domain>::TypeDom, DomainError> {
//         if self.is_in(item) {
//             let idx: usize;
//             if *item < self.upper() {
//                 idx = (((*item - self.lower()) as f64) / (self.range() as f64)
//                     * (target.values().len() as f64)) as usize;
//             } else {
//                 idx = target.values().len() - 1;
//             }
//             Ok(target.values()[idx])
//         } else {
//             Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             })
//         }
//     }
// }

// impl Convertible for Bool {
//     /// See [`Convertible`] for more infos.
//     /// Considering $(l_1,u1)$ the bounds of the [`Real`] `target` domain.
//     /// If `true` then return `target.upper()`, otherwise `target.lower()`.
//     fn to_real(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Real,
//     ) -> Result<<Real as Domain>::TypeDom, DomainError> {
//         if *item {
//             Ok(target.upper())
//         } else {
//             Ok(target.lower())
//         }
//     }
//     /// See [`Convertible`] for more infos.
//     /// Considering $(l_1,u1)$ the bounds of the [`Nat`] `target` domain.
//     /// If `true` then return `target.upper()`, otherwise `target.lower()`.
//     fn to_nat(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Nat,
//     ) -> Result<<Nat as Domain>::TypeDom, DomainError> {
//         if *item {
//             Ok(target.upper())
//         } else {
//             Ok(target.lower())
//         }
//     }
//     /// See [`Convertible`] for more infos.
//     /// Considering $(l_1,u1)$ the bounds of the [`Int`] `target` domain.
//     /// If `true` then return `target.upper()`, otherwise `target.lower()`.
//     fn to_int(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Int,
//     ) -> Result<<Int as Domain>::TypeDom, DomainError> {
//         if *item {
//             Ok(target.upper())
//         } else {
//             Ok(target.lower())
//         }
//     }

//     /// See [`Convertible`] for more infos.
//     /// Returns `*item`.
//     fn to_bool(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         _target: &Bool,
//     ) -> Result<<Bool as Domain>::TypeDom, DomainError> {
//         Ok(*item)
//     }

//     /// See [`Converter`] for more infos.
//     /// If `false` selects the first element of `values` from [`Cat`].
//     /// If `true` selects the last elements of `values` from [`Cat`].
//     ///
//     fn to_cat<'a, const N: usize>(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Cat<'a, N>,
//     ) -> Result<<Cat<'a, N> as Domain>::TypeDom, DomainError> {
//         if *item {
//             Ok(target.values()[target.values().len() - 1])
//         } else {
//             Ok(target.values()[0])
//         }
//     }
// }

// impl<'a, const N: usize> Convertible for Cat<'a, N> {
//     /// See [`Convertible`] for more infos.
//     /// Convert the index of the given item within the `values` of [`Cat`], into a float.
//     /// If $i$ is the index of `item` within `values`, $(l,u)$ the upper and lower bounds of [`Real`], then it returns:
//     /// $$ i \times (u-l) + l$$
//     fn to_real(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Real,
//     ) -> Result<<Real as Domain>::TypeDom, DomainError> {
//         let index = self.values().iter().position(|&r| r == *item);

//         if index.is_none() {
//             Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             })
//         } else {
//             let mapped = (index.unwrap() as <Real as Domain>::TypeDom)
//                 / ((self.values().len() - 1) as <Real as Domain>::TypeDom)
//                 * (target.range() as <Real as Domain>::TypeDom);
//             let mapped = mapped as <Real as Domain>::TypeDom + target.lower();

//             if target.is_in(&mapped) {
//                 Ok(mapped)
//             } else {
//                 Err(DomainError {
//                     code: 101,
//                     msg: format!("{} -> {} not in {}", item, mapped, target),
//                 })
//             }
//         }
//     }
//     /// See [`Convertible`] for more infos.
//     /// Convert the index of the given item within the `values` of [`Cat`], into a nat.
//     /// If $i$ is the index of `item` within `values`, $(l,u)$ the upper and lower bounds of [`Nat`], then it returns:
//     /// $$ i \times (u-l) + l$$
//     fn to_nat(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Nat,
//     ) -> Result<<Nat as Domain>::TypeDom, DomainError> {
//         let index = self.values().iter().position(|&r| r == *item);

//         if index.is_none() {
//             Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             })
//         } else {
//             let mapped = (index.unwrap() as <Real as Domain>::TypeDom)
//                 / ((self.values().len() - 1) as <Real as Domain>::TypeDom)
//                 * (target.range() as <Real as Domain>::TypeDom);
//             let mapped = mapped as <Nat as Domain>::TypeDom + target.lower();

//             if target.is_in(&mapped) {
//                 Ok(mapped)
//             } else {
//                 Err(DomainError {
//                     code: 101,
//                     msg: format!("{} -> {} not in {}", item, mapped, target),
//                 })
//             }
//         }
//     }
//     /// See [`Convertible`] for more infos.
//     /// Convert the index of the given item within the `values` of [`Cat`], into an int.
//     /// If $i$ is the index of `item` within `values`, $(l,u)$ the upper and lower bounds of [`Int`], then it returns:
//     /// $$ i \times (u-l) + l$$
//     fn to_int(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         target: &Int,
//     ) -> Result<<Int as Domain>::TypeDom, DomainError> {
//         let index = self.values().iter().position(|&r| r == *item);

//         if index.is_none() {
//             Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             })
//         } else {
//             let mapped = (index.unwrap() as <Real as Domain>::TypeDom)
//                 / ((self.values().len() - 1) as <Real as Domain>::TypeDom)
//                 * (target.range() as <Real as Domain>::TypeDom);
//             let mapped = mapped as <Int as Domain>::TypeDom + target.lower();

//             if target.is_in(&mapped) {
//                 Ok(mapped)
//             } else {
//                 Err(DomainError {
//                     code: 101,
//                     msg: format!("{} -> {} not in {}", item, mapped, target),
//                 })
//             }
//         }
//     }
//     /// See [`Convertible`] for more infos.
//     /// Convert the index of the given item within the `values` of [`Cat`], into a float.
//     /// If $i$ is the index of `item` within `values`, $(l,u)$ the upper and lower bounds of [`Int`], then it returns:
//     /// $$ i \times (u-l) + l$$
//     fn to_bool(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         _target: &Bool,
//     ) -> Result<<Bool as Domain>::TypeDom, DomainError> {
//         let index = self.values().iter().position(|&r| r == *item);
//         if index.is_none() {
//             Err(DomainError {
//                 code: 103,
//                 msg: format!("{} not in {}", item, self),
//             })
//         } else {
//             Ok(index.unwrap() > self.values().len() / 2)
//         }
//     }

//     /// See [`Converter`] for more infos.
//     /// ???
//     fn to_cat<'b, const M: usize>(
//         &self,
//         item: &<Self as Domain>::TypeDom,
//         _target: &Cat<'b, M>,
//     ) -> Result<<Cat<'b, M> as Domain>::TypeDom, DomainError> {
//         // <?o!o?>
//         Err(DomainError {
//             code: 101,
//             msg: format!("{} not in {}", item, self),
//         })
//     }
// }
