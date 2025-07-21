VAR TYPES : Some(Ident { ident: "Real", span: #10512 bytes(59234..59926) }) != Some(Ident { ident: "Real", span: #10512 bytes(59234..59926) }) | false
VAR TYPES : Some(Ident { ident: "Real", span: #10512 bytes(59234..59926) }) != Some(Ident { ident: "Real", span: #10512 bytes(59234..59926) }) | false
VAR TYPES : Some(Ident { ident: "Real", span: #10512 bytes(59234..59926) }) != Some(Ident { ident: "Bool", span: #10512 bytes(59234..59926) }) | true
VAR TYPES : Some(Ident { ident: "Bool", span: #10512 bytes(59234..59926) }) != Some(Ident { ident: "Nat", span: #10512 bytes(59234..59926) }) | true
IS IT MIXED true
IS IT MIXED true
IS IT MIXED true
OTHER TOKENS : let
OTHER TOKENS : a
OTHER TOKENS : =
GROUP TOKENS : [! a | Real(0.0,5.0) | !],! a | Real(0.0,5.0) | !
OTHER TOKENS : ;
OTHER TOKENS : let
OTHER TOKENS : aa
OTHER TOKENS : =
GROUP TOKENS : [! aa_{ 10 } | Real(-5.0,0.0) | !],! aa_{ 10 } | Real(-5.0,0.0) | !
OTHER TOKENS : ;
OTHER TOKENS : let
OTHER TOKENS : aaa
OTHER TOKENS : =
GROUP TOKENS : [! aaa | Real(100.0,200.0) | !],! aaa | Real(100.0,200.0) | !
OTHER TOKENS : ;
OTHER TOKENS : let
OTHER TOKENS : some_bool
OTHER TOKENS : =
GROUP TOKENS : [! boolvar | Bool() | !],! boolvar | Bool() | !
OTHER TOKENS : ;
OTHER TOKENS : let
OTHER TOKENS : some_nat
OTHER TOKENS : =
GROUP TOKENS : [! natvar | Nat(0,10) | !],! natvar | Nat(0,10) | !
OTHER TOKENS : ;
OTHER TOKENS : let
OTHER TOKENS : some_int
OTHER TOKENS : =
OTHER TOKENS : plus_one_int
GROUP TOKENS : ([! intvar | Int(-10,0) | !]),[! intvar | Int(-10,0) | !]
GROUP TOKENS : [! intvar | Int(-10,0) | !],! intvar | Int(-10,0) | !
OTHER TOKENS : ;
OTHER TOKENS : OutExample
GROUP TOKENS : {
    obj: a, fid: *aa[0], con: *aa[1], more: *aa[2], info: aaa, intinfo:
    some_int, boolinfo: some_bool, natinfo: some_nat,
},obj: a, fid: *aa[0], con: *aa[1], more: *aa[2], info: aaa, intinfo: some_int,
boolinfo: some_bool, natinfo: some_nat,
OTHER TOKENS : obj
OTHER TOKENS : :
OTHER TOKENS : a
OTHER TOKENS : ,
OTHER TOKENS : fid
OTHER TOKENS : :
OTHER TOKENS : *
OTHER TOKENS : aa
GROUP TOKENS : [0],0
OTHER TOKENS : ,
OTHER TOKENS : con
OTHER TOKENS : :
OTHER TOKENS : *
OTHER TOKENS : aa
GROUP TOKENS : [1],1
OTHER TOKENS : ,
OTHER TOKENS : more
OTHER TOKENS : :
OTHER TOKENS : *
OTHER TOKENS : aa
GROUP TOKENS : [2],2
OTHER TOKENS : ,
OTHER TOKENS : info
OTHER TOKENS : :
OTHER TOKENS : aaa
OTHER TOKENS : ,
OTHER TOKENS : intinfo
OTHER TOKENS : :
OTHER TOKENS : some_int
OTHER TOKENS : ,
OTHER TOKENS : boolinfo
OTHER TOKENS : :
OTHER TOKENS : some_bool
OTHER TOKENS : ,
OTHER TOKENS : natinfo
OTHER TOKENS : :
OTHER TOKENS : some_nat
OTHER TOKENS : ,
TOKENS : use tantale_core :: domain :: { Domain, onto :: Onto };
#[derive(tantale :: Mixed, Clone, PartialEq)] pub enum _TantaleMixedObj
{ Bool(Bool), Int(Int), Nat(Nat), Real(Real) } pub fn
_tantale__TantaleMixedObj_Nat_default_samp(dom : & _TantaleMixedObj, rng : &
mut rand :: prelude :: ThreadRng) -> < _TantaleMixedObj as tantale_core ::
domain :: Domain > :: TypeDom
{
    match dom
    {
        _TantaleMixedObj :: Nat(d) => _TantaleMixedObjTypeDom ::
        Nat(< Nat as tantale_core :: domain :: Domain > :: sample(d, rng)), _
        => unreachable!
        ("An error occured while sampling from a mixed domain. The mixed variant is of wrong type."),
    }
} pub fn
_tantale__TantaleMixedObj_Int_default_samp(dom : & _TantaleMixedObj, rng : &
mut rand :: prelude :: ThreadRng) -> < _TantaleMixedObj as tantale_core ::
domain :: Domain > :: TypeDom
{
    match dom
    {
        _TantaleMixedObj :: Int(d) => _TantaleMixedObjTypeDom ::
        Int(< Int as tantale_core :: domain :: Domain > :: sample(d, rng)), _
        => unreachable!
        ("An error occured while sampling from a mixed domain. The mixed variant is of wrong type."),
    }
} pub fn
_tantale__TantaleMixedObj_Real_default_samp(dom : & _TantaleMixedObj, rng : &
mut rand :: prelude :: ThreadRng) -> < _TantaleMixedObj as tantale_core ::
domain :: Domain > :: TypeDom
{
    match dom
    {
        _TantaleMixedObj :: Real(d) => _TantaleMixedObjTypeDom ::
        Real(< Real as tantale_core :: domain :: Domain > :: sample(d, rng)),
        _ => unreachable!
        ("An error occured while sampling from a mixed domain. The mixed variant is of wrong type."),
    }
} pub fn
_tantale__TantaleMixedObj_Bool_default_samp(dom : & _TantaleMixedObj, rng : &
mut rand :: prelude :: ThreadRng) -> < _TantaleMixedObj as tantale_core ::
domain :: Domain > :: TypeDom
{
    match dom
    {
        _TantaleMixedObj :: Bool(d) => _TantaleMixedObjTypeDom ::
        Bool(< Bool as tantale_core :: domain :: Domain > :: sample(d, rng)),
        _ => unreachable!
        ("An error occured while sampling from a mixed domain. The mixed variant is of wrong type."),
    }
} pub fn
_tantale__TantaleMixedObj_Bool_onto__TantaleMixedObj_Bool(indom : &
_TantaleMixedObj, sample : & < _TantaleMixedObj as tantale_core :: domain ::
Domain > :: TypeDom, outdom : & _TantaleMixedObj) -> Result <<
_TantaleMixedObj as tantale_core :: domain :: Domain > :: TypeDom,
tantale_core :: domain :: derrors :: DomainError > { Ok(sample.clone()) } pub
fn
_tantale__TantaleMixedObj_Nat_onto__TantaleMixedObj_Nat(indom : &
_TantaleMixedObj, sample : & < _TantaleMixedObj as tantale_core :: domain ::
Domain > :: TypeDom, outdom : & _TantaleMixedObj) -> Result <<
_TantaleMixedObj as tantale_core :: domain :: Domain > :: TypeDom,
tantale_core :: domain :: derrors :: DomainError > { Ok(sample.clone()) } pub
fn
_tantale__TantaleMixedObj_Int_onto__TantaleMixedObj_Int(indom : &
_TantaleMixedObj, sample : & < _TantaleMixedObj as tantale_core :: domain ::
Domain > :: TypeDom, outdom : & _TantaleMixedObj) -> Result <<
_TantaleMixedObj as tantale_core :: domain :: Domain > :: TypeDom,
tantale_core :: domain :: derrors :: DomainError > { Ok(sample.clone()) } pub
fn
_tantale__TantaleMixedObj_Real_onto__TantaleMixedObj_Real(indom : &
_TantaleMixedObj, sample : & < _TantaleMixedObj as tantale_core :: domain ::
Domain > :: TypeDom, outdom : & _TantaleMixedObj) -> Result <<
_TantaleMixedObj as tantale_core :: domain :: Domain > :: TypeDom,
tantale_core :: domain :: derrors :: DomainError > { Ok(sample.clone()) } pub
fn get_searchspace() -> tantale_core :: searchspace :: Sp < _TantaleMixedObj,
_TantaleMixedObj >
{
    pub use tantale_core :: domain :: { Onto, Domain }; let mut variables :
    Vec < tantale_core :: variable :: var :: Var < _TantaleMixedObj,
    _TantaleMixedObj >> = Vec :: new();
    variables.push({
        let name = ("a", None); let domobj_rc = std :: sync :: Arc ::
        new(_TantaleMixedObj :: Real(Real :: new(0.0, 5.0))); let domopt_rc =
        domobj_rc.clone(); let sampler_obj =
        _tantale__TantaleMixedObj_Real_default_samp; let sampler_opt =
        _tantale__TantaleMixedObj_Real_default_samp; let onto_obj =
        _tantale__TantaleMixedObj_Real_onto__TantaleMixedObj_Real; let
        onto_opt = _tantale__TantaleMixedObj_Real_onto__TantaleMixedObj_Real;
        let var = tantale_core :: variable :: var :: Var ::
        _new(name, domobj_rc, domopt_rc, sampler_obj, sampler_opt, onto_obj,
        onto_opt); var
    }); let mut replicates =
    {
        let name = ("aa_", None); let domobj_rc = std :: sync :: Arc ::
        new(_TantaleMixedObj :: Real(Real :: new(- 5.0, 0.0))); let domopt_rc
        = domobj_rc.clone(); let sampler_obj =
        _tantale__TantaleMixedObj_Real_default_samp; let sampler_opt =
        _tantale__TantaleMixedObj_Real_default_samp; let onto_obj =
        _tantale__TantaleMixedObj_Real_onto__TantaleMixedObj_Real; let
        onto_opt = _tantale__TantaleMixedObj_Real_onto__TantaleMixedObj_Real;
        let var = tantale_core :: variable :: var :: Var ::
        _new(name, domobj_rc, domopt_rc, sampler_obj, sampler_opt, onto_obj,
        onto_opt); var
    }.replicate(10usize); variables.append(& mut replicates);
    variables.push({
        let name = ("aaa", None); let domobj_rc = std :: sync :: Arc ::
        new(_TantaleMixedObj :: Real(Real :: new(100.0, 200.0))); let
        domopt_rc = domobj_rc.clone(); let sampler_obj =
        _tantale__TantaleMixedObj_Real_default_samp; let sampler_opt =
        _tantale__TantaleMixedObj_Real_default_samp; let onto_obj =
        _tantale__TantaleMixedObj_Real_onto__TantaleMixedObj_Real; let
        onto_opt = _tantale__TantaleMixedObj_Real_onto__TantaleMixedObj_Real;
        let var = tantale_core :: variable :: var :: Var ::
        _new(name, domobj_rc, domopt_rc, sampler_obj, sampler_opt, onto_obj,
        onto_opt); var
    });
    variables.push({
        let name = ("boolvar", None); let domobj_rc = std :: sync :: Arc ::
        new(_TantaleMixedObj :: Bool(Bool :: new())); let domopt_rc =
        domobj_rc.clone(); let sampler_obj =
        _tantale__TantaleMixedObj_Bool_default_samp; let sampler_opt =
        _tantale__TantaleMixedObj_Bool_default_samp; let onto_obj =
        _tantale__TantaleMixedObj_Bool_onto__TantaleMixedObj_Bool; let
        onto_opt = _tantale__TantaleMixedObj_Bool_onto__TantaleMixedObj_Bool;
        let var = tantale_core :: variable :: var :: Var ::
        _new(name, domobj_rc, domopt_rc, sampler_obj, sampler_opt, onto_obj,
        onto_opt); var
    });
    variables.push({
        let name = ("natvar", None); let domobj_rc = std :: sync :: Arc ::
        new(_TantaleMixedObj :: Nat(Nat :: new(0, 10))); let domopt_rc =
        domobj_rc.clone(); let sampler_obj =
        _tantale__TantaleMixedObj_Nat_default_samp; let sampler_opt =
        _tantale__TantaleMixedObj_Nat_default_samp; let onto_obj =
        _tantale__TantaleMixedObj_Nat_onto__TantaleMixedObj_Nat; let onto_opt
        = _tantale__TantaleMixedObj_Nat_onto__TantaleMixedObj_Nat; let var =
        tantale_core :: variable :: var :: Var ::
        _new(name, domobj_rc, domopt_rc, sampler_obj, sampler_opt, onto_obj,
        onto_opt); var
    });
    variables.push({
        let name = ("intvar", None); let domobj_rc = std :: sync :: Arc ::
        new(_TantaleMixedObj :: Int(Int :: new(- 10, 0))); let domopt_rc =
        domobj_rc.clone(); let sampler_obj =
        _tantale__TantaleMixedObj_Int_default_samp; let sampler_opt =
        _tantale__TantaleMixedObj_Int_default_samp; let onto_obj =
        _tantale__TantaleMixedObj_Int_onto__TantaleMixedObj_Int; let onto_opt
        = _tantale__TantaleMixedObj_Int_onto__TantaleMixedObj_Int; let var =
        tantale_core :: variable :: var :: Var ::
        _new(name, domobj_rc, domopt_rc, sampler_obj, sampler_opt, onto_obj,
        onto_opt); var
    }); tantale_core :: searchspace :: Sp
    { variables : variables.into_boxed_slice(), }
} pub fn example(tantale_in : Vec < _TantaleMixedObjTypeDom >) -> OutExample
{
    let a =
    {
        match tantale_in [0usize]
        { _TantaleMixedObjTypeDom :: Real(value) => value, _ => panic! ("") }
    }; let aa =
    {
        tantale_in
        [1usize ..
        11usize].iter().map(| v |
        {
            match v
            {
                _TantaleMixedObjTypeDom :: Real(value) => value, _ => panic!
                ("")
            }
        }).collect :: < Vec < & < Real as Domain > :: TypeDom >> ()
    }; let aaa =
    {
        match tantale_in [11usize]
        { _TantaleMixedObjTypeDom :: Real(value) => value, _ => panic! ("") }
    }; let some_bool =
    {
        match tantale_in [12usize]
        { _TantaleMixedObjTypeDom :: Bool(value) => value, _ => panic! ("") }
    }; let some_nat =
    {
        match tantale_in [13usize]
        { _TantaleMixedObjTypeDom :: Nat(value) => value, _ => panic! ("") }
    }; let some_int =
    plus_one_int({
        match tantale_in [14usize]
        { _TantaleMixedObjTypeDom :: Int(value) => value, _ => panic! ("") }
    }); OutExample
    {
        obj : a, fid : * aa [0], con : * aa [1], more : * aa [2], info : aaa,
        intinfo : some_int, boolinfo : some_bool, natinfo : some_nat,
    }
}
#[rustc_test_marker = "objective::obj_test"]
#[doc(hidden)]
pub const obj_test: test::TestDescAndFn = test::TestDescAndFn {
    desc: test::TestDesc {
        name: test::StaticTestName("objective::obj_test"),
        ignore: false,
        ignore_message: ::core::option::Option::None,
        source_file: "test_suite/tests/macros/objective.rs",
        start_line: 50usize,
        start_col: 4usize,
        end_line: 50usize,
        end_col: 12usize,
        compile_fail: false,
        no_run: false,
        should_panic: test::ShouldPanic::No,
        test_type: test::TestType::Unknown,
    },
    testfn: test::StaticTestFn(#[coverage(off)] || test::assert_test_result(obj_test())),
};
fn obj_test() {
    let sp = searchspace::get_searchspace();
}
