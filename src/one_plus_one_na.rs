use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct Network<const N: usize> {
    angles: [f64; N],
    biases: [f64; N],
}

fn generate_random_array<const N: usize>() -> [f64; N] {
    let mut array = [0.; N];
    for i in 0..N {
        array[i] = random::<f64>();
    }
    array
}

impl<const N: usize> Network<N> {
    pub fn new() -> Self {
        Self {
            angles: generate_random_array(),
            biases: generate_random_array(),
        }
    }

    pub fn optimize(&mut self, evaluation_function: fn(&Network<N>) -> f64, n_iters: u32) {
        for _ in 0..n_iters {
            let mut new_network = self.clone();

            for i in 0..N {
                if random::<f64>() < 1. / (2. * N as f64) {
                    let sign = if random::<f64>() < 0.5 { 1. } else { -1. };
                    new_network.angles[i] += sign * random::<f64>();
                    new_network.biases[i] += sign * random::<f64>();
                    new_network.angles[i] -= new_network.angles[i].floor();
                    new_network.biases[i] -= new_network.biases[i].floor();
                }
            }

            println!("{:?}", new_network);

            if evaluation_function(&new_network) > evaluation_function(self) {
                *self = new_network;
            }
        }
    }
}
