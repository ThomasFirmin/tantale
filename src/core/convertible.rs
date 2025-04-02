use crate::core::domain::{Bool, Cat, Domain, Int, Nat, NumericallyBounded, Real};
use crate::core::errors::DomainError;

pub trait Convertible {
    /// Map an input `item` from the current domain to a given [`Real`] `domain`.
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
    fn to_real(
        &self,
        _item: &<Self as Domain>::TypeDom,
        _target: &Real,
    ) -> Result<<Real as Domain>::TypeDom, DomainError>
    where
        Self: Domain,
    {
        Err(DomainError {
            code: 101,
            msg: format!(
                "Consider implement to_real method for {}",
                std::any::type_name::<Self>()
            ),
        })
    }
    /// Map an input `item` from the current domain to a given [`Nat`] `domain`.
    ///
    /// # Parameters
    ///
    /// * `item` : <Self as Domain>::TypeDom - The point from the [`Self`] domain to map to the `target` [`Nat`] domain.
    /// * `target` : &Nat - Reference to the targeted domain.
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn to_nat(
        &self,
        _item: &<Self as Domain>::TypeDom,
        _target: &Nat,
    ) -> Result<<Nat as Domain>::TypeDom, DomainError>
    where
        Self: Domain,
    {
        Err(DomainError {
            code: 101,
            msg: format!(
                "Consider implement to_nat method for {}",
                std::any::type_name::<Self>()
            ),
        })
    }
    /// Map an input `item` from the current domain to a given [`Int`] `domain`.
    ///
    /// # Parameters
    ///
    /// * `item` : <Self as Domain>::TypeDom - The point from the [`Self`] domain to map to the `target` [`Int`] domain.
    /// * `target` : &Int - Reference to the targeted domain.
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///    
    fn to_int(
        &self,
        _item: &<Self as Domain>::TypeDom,
        _target: &Int,
    ) -> Result<<Int as Domain>::TypeDom, DomainError>
    where
        Self: Domain,
    {
        Err(DomainError {
            code: 101,
            msg: format!(
                "Consider implement to_int method for {}",
                std::any::type_name::<Self>()
            ),
        })
    }

    /// Map an input `item` from the current domain to a given [`Bool`] `domain`.
    ///
    /// # Parameters
    ///
    /// * `item` : <Self as Domain>::TypeDom - The point from the [`Self`] domain to map to the `target` [`Bool`] domain.
    /// * `target` : &Bool - Reference to the targeted domain.
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn to_bool(
        &self,
        _item: &<Self as Domain>::TypeDom,
        _target: &Bool,
    ) -> Result<<Bool as Domain>::TypeDom, DomainError>
    where
        Self: Domain,
    {
        Err(DomainError {
            code: 101,
            msg: format!(
                "Consider implement to_bool method for {}",
                std::any::type_name::<Self>()
            ),
        })
    }

    /// Map an input `item` from the current domain to a given [`Cat`] `domain`.
    ///
    /// # Parameters
    ///
    /// * `item` : <Self as Domain>::TypeDom - The point from the [`Self`] domain to map to the `target` [`Cat`] domain.
    /// * `target` : &Cat<'a,N> - Reference to the targeted domain.
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn to_cat<'a, const N: usize>(
        &self,
        _item: &<Self as Domain>::TypeDom,
        _target: &Cat<'a, N>,
    ) -> Result<<Cat<'a, N> as Domain>::TypeDom, DomainError>
    where
        Self: Domain,
    {
        Err(DomainError {
            code: 101,
            msg: format!(
                "Consider implement to_real method for {}",
                std::any::type_name::<Self>()
            ),
        })
    }
}

impl Convertible for Real {
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Real`] domain, the mapping is given by:
    /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
    fn to_real(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Real,
    ) -> Result<<Real as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom) / (self.range() as <Real as Domain>::TypeDom);
            let mapped = normalized * target.range() + target.lower();

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
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Nat`] domain, the mapping is given by:
    /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
    fn to_nat(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Nat,
    ) -> Result<<Nat as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom) / (self.range() as <Real as Domain>::TypeDom);
            let mapped = (normalized * (target.range() as <Real as Domain>::TypeDom))as <Nat as Domain>::TypeDom + target.lower();

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
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Int`] domain, the mapping is given by:
    /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
    fn to_int(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Int,
    ) -> Result<<Int as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom) / (self.range() as <Real as Domain>::TypeDom);
            let mapped = (normalized * (target.range() as <Real as Domain>::TypeDom))as <Int as Domain>::TypeDom + target.lower();

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
    /// See [`Convertible`] for more infos.
    /// Considering $m$ as the middle point of the [`Real`] domain, the mapping is given by:
    /// $$\texttt{item} > m$$
    fn to_bool(
        &self,
        item: &<Self as Domain>::TypeDom,
        _target: &Bool,
    ) -> Result<<Bool as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            Ok(*item > self.mid())
        } else {
            return Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            });
        }
    }

    /// See [`Converter`] for more infos.
    /// Maps a float to an index of
    fn to_cat<'a, const N: usize>(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Cat<'a, N>,
    ) -> Result<<Cat<'a, N> as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let idx: usize;
            if *item < self.upper() {
                idx = ((*item - self.lower()) / self.range()
                    * target.values().len() as <Self as Domain>::TypeDom)
                    .floor() as usize;
            } else {
                idx = target.values().len() - 1;
            }
            Ok(target.values()[idx])
        } else {
            Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            })
        }
    }
}

impl Convertible for Nat {
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Real`] domain, the mapping is given by:
    /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
    fn to_real(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Real,
    ) -> Result<<Real as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom) / (self.range() as <Real as Domain>::TypeDom);
            let mapped = normalized * target.range() + target.lower();

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
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Nat`] domain, the mapping is given by:
    /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
    fn to_nat(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Nat,
    ) -> Result<<Nat as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom) / (self.range() as <Real as Domain>::TypeDom);
            let mapped = (normalized * (target.range() as <Real as Domain>::TypeDom))as <Nat as Domain>::TypeDom + target.lower();

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
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Int`] domain, the mapping is given by:
    /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
    fn to_int(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Int,
    ) -> Result<<Int as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom) / (self.range() as <Real as Domain>::TypeDom);
            let mapped = (normalized * (target.range() as <Real as Domain>::TypeDom))as <Int as Domain>::TypeDom + target.lower();

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
    /// See [`Convertible`] for more infos.
    /// Considering $m$ as the middle point of the [`Real`] domain, the mapping is given by:
    /// $$\texttt{item} > m$$
    fn to_bool(
        &self,
        item: &<Self as Domain>::TypeDom,
        _target: &Bool,
    ) -> Result<<Bool as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            Ok(*item > self.mid())
        } else {
            return Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            });
        }
    }

    /// See [`Converter`] for more infos.
    /// Maps a float to an index of
    fn to_cat<'a, const N: usize>(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Cat<'a, N>,
    ) -> Result<<Cat<'a, N> as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let idx: usize;
            if *item < self.upper() {
                idx = ((*item - self.lower()) / self.range()
                    * target.values().len() as <Self as Domain>::TypeDom)
                    as usize;
            } else {
                idx = target.values().len() - 1;
            }
            Ok(target.values()[idx])
        } else {
            Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            })
        }
    }
}

impl Convertible for Int {
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Real`] domain, the mapping is given by:
    /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
    fn to_real(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Real,
    ) -> Result<<Real as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom) / (self.range() as <Real as Domain>::TypeDom);
            let mapped = normalized * target.range() + target.lower();

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
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Nat`] domain, the mapping is given by:
    /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
    fn to_nat(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Nat,
    ) -> Result<<Nat as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom) / (self.range() as <Real as Domain>::TypeDom);
            let mapped = (normalized * (target.range() as <Real as Domain>::TypeDom))as <Nat as Domain>::TypeDom + target.lower();

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
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Int`] domain, the mapping is given by:
    /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
    fn to_int(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Int,
    ) -> Result<<Int as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom) / (self.range() as <Real as Domain>::TypeDom);
            let mapped = (normalized * (target.range() as <Real as Domain>::TypeDom))as <Int as Domain>::TypeDom + target.lower();

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
    /// See [`Convertible`] for more infos.
    /// Considering $m$ as the middle point of the [`Real`] domain, the mapping is given by:
    /// $$\texttt{item} > m$$
    fn to_bool(
        &self,
        item: &<Self as Domain>::TypeDom,
        _target: &Bool,
    ) -> Result<<Bool as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            Ok(*item > self.mid())
        } else {
            return Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            });
        }
    }

    /// See [`Converter`] for more infos.
    /// Maps a float to an index of
    fn to_cat<'a, const N: usize>(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Cat<'a, N>,
    ) -> Result<<Cat<'a, N> as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let idx: usize;
            if *item < self.upper() {
                idx = ((*item - self.lower()) / self.range()
                    * target.values().len() as <Self as Domain>::TypeDom)
                    as usize;
            } else {
                idx = target.values().len() - 1;
            }
            Ok(target.values()[idx])
        } else {
            Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            })
        }
    }
}

impl Convertible for Bool {
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] `target` domain.
    /// If `true` then return `target.upper()`, otherwise `target.lower()`.
    fn to_real(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Real,
    ) -> Result<<Real as Domain>::TypeDom, DomainError> {
        if *item {
            Ok(target.upper())
        } else {
            Ok(target.lower())
        }
    }
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Nat`] `target` domain.
    /// If `true` then return `target.upper()`, otherwise `target.lower()`.
    fn to_nat(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Nat,
    ) -> Result<<Nat as Domain>::TypeDom, DomainError> {
        if *item {
            Ok(target.upper())
        } else {
            Ok(target.lower())
        }
    }
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Int`] `target` domain.
    /// If `true` then return `target.upper()`, otherwise `target.lower()`.
    fn to_int(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Int,
    ) -> Result<<Int as Domain>::TypeDom, DomainError> {
        if *item {
            Ok(target.upper())
        } else {
            Ok(target.lower())
        }
    }

    /// See [`Convertible`] for more infos.
    /// Returns `*item`.
    fn to_bool(
        &self,
        item: &<Self as Domain>::TypeDom,
        _target: &Bool,
    ) -> Result<<Bool as Domain>::TypeDom, DomainError> {
        Ok(*item)
    }
}


impl <'a,const N:usize>Convertible for Cat<'a,N> {
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Real`] domain, the mapping is given by:
    /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
    fn to_real(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Real,
    ) -> Result<<Real as Domain>::TypeDom, DomainError> {
        let index = self.values().iter().position(|&r| r==*item);

        if index.is_none(){
            Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            })
        }else{
            let mapped = index.unwrap();

        }

        if target.is_in(&mapped) {
            Ok(mapped)
        } else {
            Err(DomainError {
                code: 101,
                msg: format!("{} -> {} not in {}", item, mapped, target),
            })
        }
    }
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Nat`] domain, the mapping is given by:
    /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
    fn to_nat(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Nat,
    ) -> Result<<Nat as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom) / (self.range() as <Real as Domain>::TypeDom);
            let mapped = (normalized * (target.range() as <Real as Domain>::TypeDom))as <Nat as Domain>::TypeDom + target.lower();

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
    /// See [`Convertible`] for more infos.
    /// Considering $(l_1,u1)$ the bounds of the [`Real`] domain, and $(l_2,u_2)$ the bounds of the [`Int`] domain, the mapping is given by:
    /// $$\frac{x - l_1}{u_1 - l_1} \times (u_2 - l_2) + l_2$$
    fn to_int(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Int,
    ) -> Result<<Int as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let normalized = ((*item - self.lower()) as <Real as Domain>::TypeDom) / (self.range() as <Real as Domain>::TypeDom);
            let mapped = (normalized * (target.range() as <Real as Domain>::TypeDom))as <Int as Domain>::TypeDom + target.lower();

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
    /// See [`Convertible`] for more infos.
    /// Considering $m$ as the middle point of the [`Real`] domain, the mapping is given by:
    /// $$\texttt{item} > m$$
    fn to_bool(
        &self,
        item: &<Self as Domain>::TypeDom,
        _target: &Bool,
    ) -> Result<<Bool as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            Ok(*item > self.mid())
        } else {
            return Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            });
        }
    }

    /// See [`Converter`] for more infos.
    /// Maps a float to an index of
    fn to_cat<'a, const N: usize>(
        &self,
        item: &<Self as Domain>::TypeDom,
        target: &Cat<'a, N>,
    ) -> Result<<Cat<'a, N> as Domain>::TypeDom, DomainError> {
        if self.is_in(item) {
            let idx: usize;
            if *item < self.upper() {
                idx = ((*item - self.lower()) / self.range()
                    * target.values().len() as <Self as Domain>::TypeDom)
                    .floor() as usize;
            } else {
                idx = target.values().len() - 1;
            }
            Ok(target.values()[idx])
        } else {
            Err(DomainError {
                code: 103,
                msg: format!("{} not in {}", item, self),
            })
        }
    }
}

