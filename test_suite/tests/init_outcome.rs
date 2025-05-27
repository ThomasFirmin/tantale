use tantale::core::HashOut;

pub fn get_map() -> HashOut {
    HashOut::from([
        ("obj", 42.0),
        ("other", 1.0),
        ("info", 10.0),
    ]
    )
}