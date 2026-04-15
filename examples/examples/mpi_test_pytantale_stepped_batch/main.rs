#[path = "../cleaner.rs"]
pub mod cleaner;

#[path = "../run_checker.rs"]
pub mod run_checker;

pub mod test_pytantale_fid_batch;

fn main() {
    test_pytantale_fid_batch::test_python_function();
}
