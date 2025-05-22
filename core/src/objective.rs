pub mod obj;
pub use obj::{Objective, SliceObjective, HashMapObjective, PyKwargsObjective};

pub mod criteria;
pub use criteria::{Criteria, Max, Min, Lambda};