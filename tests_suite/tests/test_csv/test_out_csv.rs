use tantale::core::saver::csvsaver::CSVWritable;

mod outcome {
    use serde::{Deserialize, Serialize};
    use tantale_macros::Outcome;

    #[derive(Outcome, Debug, Serialize, Deserialize)]
    pub struct OutExample {
        pub fid2: f64,
        pub con3: f64,
        pub con4: f64,
        pub con5: f64,
        pub mul6: f64,
        pub mul7: f64,
        pub mul8: f64,
        pub mul9: f64,
        pub tvec: Vec<f64>,
    }

    pub fn get_struct() -> OutExample {
        OutExample {
            fid2: 2.2,
            con3: 3.3,
            con4: 4.4,
            con5: 5.5,
            mul6: 6.6,
            mul7: 7.7,
            mul8: 8.8,
            mul9: 9.9,
            tvec: Vec::from([1.1, 2.2, 3.3]),
        }
    }
}

#[test]
fn out_header() {
    let head = outcome::OutExample::header(&());
    let str_true = Vec::from([
        String::from("fid2"),
        String::from("con3"),
        String::from("con4"),
        String::from("con5"),
        String::from("mul6"),
        String::from("mul7"),
        String::from("mul8"),
        String::from("mul9"),
        String::from("tvec"),
    ]);
    assert_eq!(
        head, str_true,
        "Header of outcome does not match the true baseline."
    );
}

#[test]
fn out_write() {
    let out = outcome::get_struct();
    let wrote = out.write(&());
    let str_true = Vec::from([
        String::from("2.2"),
        String::from("3.3"),
        String::from("4.4"),
        String::from("5.5"),
        String::from("6.6"),
        String::from("7.7"),
        String::from("8.8"),
        String::from("9.9"),
        String::from("[1.1, 2.2, 3.3]"),
    ]);
    assert_eq!(
        wrote, str_true,
        "Write output of outcome does not match the true baseline."
    );
}
