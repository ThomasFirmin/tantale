use tantale_core::Dominate;

use crate::mo::ParetoFront;

/// Represents the errors that can occur when computing the WFG hypervolume indicator.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum WFGError {
    ObjectivesInvalid,
}

impl std::fmt::Display for WFGError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            WFGError::ObjectivesInvalid => write!(f, "Inputs have different number of objectives"),
        }
    }
}

pub fn worse_from_two<T: Dominate, U: Dominate>(a: &T, b: &U) -> Vec<f64> {
    a.iter_obj()
        .zip(b.iter_obj())
        .map(|(x, y)| x.min(y))
        .collect()
}

pub fn wfg_limits<T: Dominate>(front: &[&T], k:usize) -> Vec<Vec<f64>>
{
    let pivot = front[k];
    front[k+1..].iter().map(
        |sol| worse_from_two(pivot, *sol)
    ).collect()
}

pub fn wfg_inclhv<T: Dominate, U: Dominate>(point: &T, ref_point: &U) -> f64
{
    point.iter_obj().zip(ref_point.iter_obj()).map(
        |(y1, y2)| (y1 - y2).abs()
    ).product()
}


pub fn wfg_exclhv<T: Dominate, U: Dominate>(front: &[&T], ref_point: &U, k:usize) ->f64
{
    let limits = wfg_limits(front, k);
    let pareto = limits.pareto_front();
    let inclusive = wfg_inclhv(front[k], ref_point);
    let wfg =  wfg_hv(&pareto, ref_point);
    inclusive - wfg
}

pub fn wfg_hv<T: Dominate, U: Dominate>(front: &[&T], ref_point: &U) -> f64
{
    (0..front.len()).map(|k| wfg_exclhv(front, ref_point, k)).sum()
}



pub fn wfg_limits_extra<T: Dominate, E: Dominate>(front: &[&T], extra: &E, k:usize) -> Vec<Vec<f64>>
{
    if k == front.len(){
        Vec::new()
    } else {
        let pivot = front[k];
        let mut res: Vec<Vec<f64>> = front[k+1..].iter().map(
            |sol| worse_from_two(pivot, *sol)
        ).collect();
        res.push(worse_from_two(pivot, extra));
        res
    }
}

pub fn wfg_exclhv_extra<T: Dominate, E:Dominate, U: Dominate>(front: &[&T], extra: &E, ref_point: &U, k:usize) ->f64
{
    let limits = wfg_limits_extra(front, extra, k);
    let pareto = limits.pareto_front();
    let inclusive = if k == front.len() {
         wfg_inclhv(extra, ref_point)
    } else {
         wfg_inclhv(front[k], ref_point)
    };
    let wfg =  wfg_hv(&pareto, ref_point);
    inclusive - wfg
}

pub fn wfg_hv_extra<T: Dominate, E:Dominate, U: Dominate>(front: &[&T], extra: &E, ref_point: &U) -> f64
{
    (0..front.len()+1).map(|k| wfg_exclhv_extra(front, extra, ref_point, k)).sum()
}