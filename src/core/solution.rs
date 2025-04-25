use crate::core::domain::Domain;


pub struct Solution<T:Domain>{
    pub id : u64,
    pub x : Vec<T::TypeDom>,
}