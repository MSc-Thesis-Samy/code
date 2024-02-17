use cmaes::DVector;
use crate::network::*;

pub fn half<const N: usize>(x: &DVector<f64>) -> f64 {
    let network: Network<N,2> = get_network(&x);
    crate::one_plus_one_na::half(&network)
}

pub fn quarter<const N: usize>(x: &DVector<f64>) -> f64 {
    let network: Network<N,2> = get_network(&x);
    crate::one_plus_one_na::quarter(&network)
}

pub fn two_quarters<const N: usize>(x: &DVector<f64>) -> f64 {
    let network: Network<N,2> = get_network(&x);
    crate::one_plus_one_na::two_quarters(&network)
}

pub fn square<const N: usize>(x: &DVector<f64>) -> f64 {
    let network: Network<N,2> = get_network(&x);
    crate::one_plus_one_na::square(&network)
}

pub fn cube<const N: usize>(x: &DVector<f64>) -> f64 {
    let network: Network<N,3> = get_network(&x);
    crate::one_plus_one_na::cube(&network)
}
