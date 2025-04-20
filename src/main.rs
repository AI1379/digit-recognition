use mnist::{Mnist, MnistBuilder};
use ndarray::{
    prelude::{Array2, Array3},
    s,
};

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("data/MNIST/raw")
        .label_format_digit()
        .training_set_length(50000)
        .validation_set_length(10000)
        .test_set_length(10000)
        .finalize();

    let image_num = 0;

    let train_data = Array3::from_shape_vec((50000, 28, 28), trn_img)
        .expect("Failed to create train data array")
        .map(|x| *x as f32 / 256.0);

    println!("Train data shape: {:?}", train_data.shape());
    println!("{:#.1?}\n", train_data.slice(s![image_num, .., ..]));

    let train_labels = Array2::from_shape_vec((50000, 1), trn_lbl)
        .expect("Failed to create train labels array")
        .map(|x| *x as f32);
    println!("Train labels shape: {:?}", train_labels.shape());
    println!(
        "The first digit is a {:?}",
        train_labels.slice(s![image_num, ..])
    );

    let test_data = Array3::from_shape_vec((10000, 28, 28), tst_img)
        .expect("Failed to create test data array")
        .map(|x| *x as f32 / 256.0);
    println!("Test data shape: {:?}", test_data.shape());
    println!("{:#.1?}\n", test_data.slice(s![image_num, .., ..]));

    let test_labels = Array2::from_shape_vec((10000, 1), tst_lbl)
        .expect("Failed to create test labels array")
        .map(|x| *x as f32);
    println!("Test labels shape: {:?}", test_labels.shape());
    println!(
        "The first digit is a {:?}",
        test_labels.slice(s![image_num, ..])
    );
}
