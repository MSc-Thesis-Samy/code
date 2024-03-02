use rand::prelude::*;
use rand_distr::Uniform;

pub fn polar_dot_product(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
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

pub fn sample_harmonic_distribution<R: Rng>(rng: &mut R, r: usize) -> u32 {
    let harmonic_sum: f64 = (1..=r).map(|i| 1.0 / (i as f64)).sum();

    let uniform = Uniform::new(0.0, 1.0);
    let rand_value = rng.sample(uniform);

    let mut sum = 0.0;
    for i in 1..=r {
        let prob = 1.0 / (i as f64 * harmonic_sum);
        sum += prob;
        if rand_value <= sum {
            return i as u32;
        }
    }

    unreachable!()
}
