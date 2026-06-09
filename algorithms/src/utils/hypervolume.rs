use tantale_core::{Dominate, HasY, Multi, Outcome, TypeCodom};

pub fn wfg_inclhv<T, Out>(point: &T, ref_point: &TypeCodom<Out>, n_obj: usize) -> f64
where 
    T : Dominate + HasY<Out>,
    Out : Outcome,
    Out::Cod: Multi<Out>,
    TypeCodom<Out>: Dominate,
{
    (0..n_obj).map(
        |idx| 
        {   
            (point.get_objective_by_index(idx) - ref_point.get_objective_by_index(idx)).abs()
        }
    ).product()
}

pub fn wfg_limits<T, Out>(front: &[T], k:usize, n_obj: usize) -> Vec<Vec<f64>>
where
    T: Dominate + HasY<Out>,
    Out: Outcome,
    TypeCodom<Out>: Dominate,
{
    let lenmk = front.len() - k;
    let pivot = &front[lenmk];
    front[..lenmk].iter().map(
        |sol|{
            (0..n_obj).map(
                |j|
                pivot.get_objective_by_index(j).min(sol.get_objective_by_index(j))
            ).collect()
        }
    ).collect()
}

pub fn wfg_exclhv<T, Out>(front: &[T], ref_point: &TypeCodom<Out>, k:usize, n_obj: usize) -> Vec<Vec<f64>>
where
    T: Dominate + HasY<Out>,
    Out: Outcome,
    TypeCodom<Out>: Dominate,
{
    let point = &front[k];
    // let inclusive = wfg_inclhv(point, ref_point, n_obj);
    let limits = wfg_limits(front, k, n_obj);
    todo!()
}