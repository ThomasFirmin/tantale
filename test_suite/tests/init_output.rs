use std::collections::HashMap;

pub fn get_map() -> HashMap<&'static str, f64> {
    HashMap::from([
        ("obj", 42.0),
        ("other", 1.0),
        ("info", 10.0),
    ]
    )
}