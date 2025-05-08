use crate::var::Var;

pub struct VarVec
{
    variables : Vec<V>,
    vlen:usize, // True length
    tlen:usize, // Virtual length
}

struct VarIter{
    var : &Var<'a, Obj, Opt>,
    varvec : &VarVec,
    pos:(usize,usize),
}

impl VarVec{
    fn iter(&self) -> VarIter{
        let first = self.variables.first().unwrap();
        VarIter { var: first, varvec:&self , pos: first }
    }
}