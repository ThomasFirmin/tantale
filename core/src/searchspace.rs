// use crate::domain::Domain;
// use crate::solution::Solution;
// use crate::variable::Var;

// use std::fmt::{Debug, Display};

// pub trait Searchspace<Obj, Opt>
// where
//     Obj: Domain + Clone + Display + Debug,
//     Opt: Domain + Clone + Display + Debug,
// {
//     fn onto_obj(&self, item: Solution<Obj>) -> Solution<Opt>;
//     fn onto_opt(&self, item: Solution<Opt>) -> Solution<Obj>;
//     fn sample_obj(&self) -> Solution<Obj>;
//     fn sample_opt(&self) -> Solution<Opt>;
// }

// pub struct Sp<Obj, Opt=Obj>
// where
//     Obj: Domain + Clone + Display + Debug,
//     Opt: Domain + Clone + Display + Debug,
// {
//     pub variables: Vec<Var<Obj, Opt>>,
// }

// impl <Obj,Opt> Searchspace<Obj,Opt> for Sp<Obj,Opt>
// where
//     Obj: Domain + Clone + Display + Debug,
//     Opt: Domain + Clone + Display + Debug,
// {
//     fn onto_obj(&self, item: Solution<Obj>) -> Solution<Opt> {
//         todo!()
//     }

//     fn onto_opt(&self, item: Solution<Opt>) -> Solution<Obj> {
//         todo!()
//     }

//     fn sample_obj(&self) -> Solution<Obj> {
//         todo!()
//     }

//     fn sample_opt(&self) -> Solution<Opt> {
//         todo!()
//     }
// }