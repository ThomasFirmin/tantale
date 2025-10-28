use rmp_serde;
use tantale::core::saver::csvsaver::CSVSaver;

#[test]
fn test_calls_json() {
    let saver = CSVSaver::new("tmp_test", true, true, true, true, 4);
    let st_ser = rmp_serde::encode::to_vec(&saver).unwrap();
    let nsaver: CSVSaver = rmp_serde::decode::from_slice(&st_ser).unwrap();

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
    assert_eq!(
        saver.save_info, nsaver.save_info,
        "Serde mismatch on loaded saver save_info"
    );
}
