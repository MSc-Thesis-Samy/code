use cmaes::DVector;
use crate::network::*;

pub fn half<const N: usize>(x: &DVector<f64>) -> f64 {
    let network: Network<N,2> = Network::get_network(&x);
    crate::oneplusone_na::half(&network)
}

pub fn quarter<const N: usize>(x: &DVector<f64>) -> f64 {
    let network: Network<N,2> = Network::get_network(&x);
    crate::oneplusone_na::quarter(&network)
}

pub fn two_quarters<const N: usize>(x: &DVector<f64>) -> f64 {
    let network: Network<N,2> = Network::get_network(&x);
    crate::oneplusone_na::two_quarters(&network)
}

pub fn square<const N: usize>(x: &DVector<f64>) -> f64 {
    let network: Network<N,2> = Network::get_network(&x);
    crate::oneplusone_na::square(&network)
}

pub fn cube<const N: usize>(x: &DVector<f64>) -> f64 {
    let network: Network<N,3> = Network::get_network(&x);
    crate::oneplusone_na::cube(&network)
}
