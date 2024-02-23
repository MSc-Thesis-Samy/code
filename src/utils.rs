use rand::prelude::*;

pub fn generate_random_array_2d<const N: usize, const D: usize>() -> [[f64;D];N] {
    let mut array = [[0.;D];N];
    for i in 0..N {
        for j in 0..D {
            array[i][j] = random::<f64>();
        }
    }
    array
}

pub fn polar_dot_product<const D: usize>(v1: &[f64; D], v2: &[f64; D]) -> f64 {
    let (r1, angles1) = v1.split_first().unwrap();
    let (r2, angles2) = v2.split_first().unwrap();

    let norm_product = r1 * r2;
    let angle_difference_cosine: f64 = angles1
        .iter()
        .zip(angles2.iter())
        .map(|(&theta1, &theta2)| (theta1 - theta2).cos())
        .product();

    norm_product * angle_difference_cosine
}

pub fn polar_dot_product_vect(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    let (r1, angles1) = v1.split_first().unwrap();
    let (r2, angles2) = v2.split_first().unwrap();

    let norm_product = r1 * r2;
    let angle_difference_cosine: f64 = angles1
        .iter()
        .zip(angles2.iter())
        .map(|(&theta1, &theta2)| (theta1 - theta2).cos())
        .product();

    norm_product * angle_difference_cosine
}
