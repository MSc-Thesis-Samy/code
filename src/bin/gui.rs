use std::fs::File;
use ggez::*;
use neuroevolution::discrete_network::DiscreteNetwork;
use neuroevolution::gui::*;
use neuroevolution::neuroevolution_algorithm::*;
use neuroevolution::constants::*;
use neuroevolution::benchmarks::*;

fn main() {
    let network = DiscreteNetwork::new(RESOLUTION, 2, 2);
    let alg = Algorithm::DiscreteOneplusoneNA(network);

    let state = State::new(alg, two_quarters, N_ITERATIONS);

    let mut conf_file = File::open("gui_conf.toml").unwrap();
    let conf = conf::Conf::from_toml_file(&mut conf_file).unwrap();

    let cb = ContextBuilder::new("Neuroevolution", "Samy Haffoudhi") .default_conf(conf);
    let (ctx, event_loop) = cb.build().unwrap();

    event::run(ctx, event_loop, state);
}
