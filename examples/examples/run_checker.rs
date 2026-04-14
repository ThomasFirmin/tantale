use std::path::Path;

pub fn run_reader(path: &str, size: usize) {
    let true_path = Path::new(path);
    let eval_path = true_path.join(Path::new("recorder")).join(Path::new("recorder_rank0"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");
    let path_info = eval_path.join("info.csv");

    // Check `Obj`, `Opt`, `Codom`
    let mut rdr_obj = csv::Reader::from_path(path_obj).unwrap();
    let mut rdr_opt = csv::Reader::from_path(path_opt).unwrap();
    let mut rdr_cod = csv::Reader::from_path(path_cod).unwrap();
    let mut rdr_info = csv::Reader::from_path(path_info).unwrap();
    let mut rdr_out = csv::Reader::from_path(path_out).unwrap();

    let linesobj = rdr_obj.records();
    let linesopt = rdr_opt.records();
    let linescod = rdr_cod.records();
    let linesinfo = rdr_info.records();
    let linesout = rdr_out.records();

    let count_obj = linesobj.count();
    let count_opt = linesopt.count();
    let count_cod = linescod.count();
    let count_info = linesinfo.count();
    let count_out = linesout.count();

    let linesobj = rdr_obj.records();
    linesobj.for_each(|l| println!("{:?}", l));
    assert_eq!(count_obj, size, "Some solutions are missing in obj.");
    assert_eq!(count_opt, size, "Some solutions are missing in opt.");
    assert_eq!(count_cod, size, "Some solutions are missing in cod.");
    assert_eq!(count_info, size, "Some solutions are missing in info.");
    assert_eq!(count_out, size, "Some solutions are missing in out.");
    assert!(
        [count_opt, count_cod, count_info, count_out]
            .iter()
            .all(|c| c == &count_obj),
        "Not all counts are equal. Some solutions are missing within at least one save file"
    );
}

pub fn run_reader_eps(path: &str, size: usize, epsilon: usize) {
    let true_path = Path::new(path);
    let eval_path = true_path.join(Path::new("recorder")).join(Path::new("recorder_rank0"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");
    let path_info = eval_path.join("info.csv");

    // Check `Obj`, `Opt`, `Codom`
    let mut rdr_obj = csv::Reader::from_path(path_obj).unwrap();
    let mut rdr_opt = csv::Reader::from_path(path_opt).unwrap();
    let mut rdr_cod = csv::Reader::from_path(path_cod).unwrap();
    let mut rdr_info = csv::Reader::from_path(path_info).unwrap();
    let mut rdr_out = csv::Reader::from_path(path_out).unwrap();

    let linesobj = rdr_obj.records();
    let linesopt = rdr_opt.records();
    let linescod = rdr_cod.records();
    let linesinfo = rdr_info.records();
    let linesout = rdr_out.records();

    let count_obj = linesobj.count();
    let count_opt = linesopt.count();
    let count_cod = linescod.count();
    let count_info = linesinfo.count();
    let count_out = linesout.count();

    let linesobj = rdr_obj.records();
    linesobj.for_each(|l| println!("{:?}", l));
    assert!(
        (count_obj >= size) && (count_obj < size + epsilon),
        "Some solutions are missing in obj {} >= {}, {} < {}.",
        count_obj,
        size,
        count_obj,
        size + epsilon
    );
    assert!(
        (count_opt >= size) && (count_opt < size + epsilon),
        "Some solutions are missing in opt {} >= {}, {} < {}.",
        count_opt,
        size,
        count_opt,
        size + epsilon
    );
    assert!(
        (count_cod >= size) && (count_cod < size + epsilon),
        "Some solutions are missing in cod {} >= {}, {} < {}.",
        count_cod,
        size,
        count_cod,
        size + epsilon
    );
    assert!(
        (count_info >= size) && (count_info < size + epsilon),
        "Some solutions are missing in info {} >= {}, {} < {}.",
        count_info,
        size,
        count_info,
        size + epsilon
    );
    assert!(
        (count_out >= size) && (count_out < size + epsilon),
        "Some solutions are missing in out {} >= {}, {} < {}.",
        count_out,
        size,
        count_out,
        size + epsilon
    );
    assert!(
        [count_opt, count_cod, count_info, count_out]
            .iter()
            .all(|c| c == &count_obj),
        "Not all counts are equal. Some solutions are missing within at least one save file"
    );
}

#[allow(dead_code)]
fn main() { }