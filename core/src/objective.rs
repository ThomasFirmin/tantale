pub mod obj;
pub use obj::{Objective, SimpleObjective};

pub mod codomain;
pub use codomain::{Criteria, Codomain, Single, Multi, Constrained, Fidelity};

pub mod outcome;
pub use outcome::{Outcome, HashOut};