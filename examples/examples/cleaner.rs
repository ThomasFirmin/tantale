pub struct Cleaner(String);

impl Cleaner {
    pub fn new(path: &str) -> Self {
        let _ = std::fs::remove_dir_all(path);
        Cleaner(String::from(path))
    }
}

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[allow(dead_code)]
fn main() {}
