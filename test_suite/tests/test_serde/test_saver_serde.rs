use serde_json;
use tantale::core::saver::csvsaver::CSVSaver;

#[test]
fn test_calls_json() {
    let saver = CSVSaver::new("tmp_test", true, true, true, 4);
    let st_ser = serde_json::to_string(&saver).unwrap();
    let nsaver: CSVSaver = serde_json::from_str(&st_ser).unwrap();

    assert_eq!(
        saver.checkpoint, nsaver.checkpoint,
        "Serde mismatch on loaded saver checkpoint"
    );
    assert_eq!(
        saver.path, nsaver.path,
        "Serde mismatch on loaded saver path"
    );
    assert_eq!(
        saver.save_obj, nsaver.save_obj,
        "Serde mismatch on loaded saver save_obj"
    );
    assert_eq!(
        saver.save_opt, nsaver.save_opt,
        "Serde mismatch on loaded saver save_opt"
    );
    assert_eq!(
        saver.save_out, nsaver.save_out,
        "Serde mismatch on loaded saver save_out"
    );
}
