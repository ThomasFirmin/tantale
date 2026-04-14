#[path = "../cleaner.rs"]
pub mod cleaner;

#[path ="../run_checker.rs"]
pub mod run_checker;

pub mod test_pytantale_batch;

pub fn main() {
    test_pytantale_batch::test_python_function();
}