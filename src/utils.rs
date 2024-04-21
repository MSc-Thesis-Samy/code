use std::fs::File;
use std::io::{BufRead, BufReader};
use rand::prelude::*;
use rand_distr::Uniform;
use crate::benchmarks::LabeledPoints;

pub fn read_cancer1_file() -> LabeledPoints {
    let file = File::open("cancer1.txt").expect("Failed to open file");
    let reader = BufReader::new(file);

    let mut data: LabeledPoints = Vec::new();
    for line in reader.lines() {
        if let Ok(line) = line {
            let mut parts = line.split_whitespace();
            let features: Vec<f64> = parts
                .by_ref()
                .take(9)
                .map(|s| s.parse().unwrap())
                .collect();
            let class = parts.next().unwrap().parse().unwrap();
            data.push((features, class));
        }
    }

    data
}

fn to_cartesian(spherical_coords: &Vec<f64>) -> Vec<f64> {
    let r = spherical_coords[0];
    let angles = &spherical_coords[1..];
    let num_dims = angles.len() + 1;

    let mut cartesian_coords = vec![0.0; num_dims];

    // Compute cartesian coordinates
    cartesian_coords[0] = r * angles[0].cos();
    let mut sin_product = angles[0].sin();
    for i in 1..num_dims - 1 {
        sin_product *= angles[i - 1].sin();
        cartesian_coords[i] = r * sin_product * angles[i].cos();
    }
    sin_product *= angles[num_dims - 2].sin();
    cartesian_coords[num_dims - 1] = r * sin_product;

    cartesian_coords
}

pub fn polar_dot_product(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    match v1.len() {
        2 => {
            let (r1, theta1) = (v1[0], v1[1]);
            let (r2, theta2) = (v2[0], v2[1]);

            r1 * r2 * (theta1 - theta2).cos()
        }
        _ => {
            let v1 = to_cartesian(v1);
            let v2 = to_cartesian(v2);
            v1.iter().zip(v2.iter()).map(|(x, y)| x * y).sum()
        }
    }
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
