use std::f64::consts::PI;
use std::fs::File;
use ggez::*;
use neuroevolution::gui::*;
use neuroevolution::network::Network;
use neuroevolution::neuroevolution_algorithm::Algorithm;

fn main() {
    let network = Network::from_parameters(
        vec![0.],
        vec![vec![PI / 2.]]
    );

    let algorithm = Algorithm::ContinuousOneplusoneNA(network);
    let state = State::new(algorithm);

    let mut conf_file = File::open("gui_conf.toml").unwrap();
    let conf = conf::Conf::from_toml_file(&mut conf_file).unwrap();

    let cb = ContextBuilder::new("Neuroevolution", "Samy Haffoudhi") .default_conf(conf);
    let (ctx, event_loop) = cb.build().unwrap();

    event::run(ctx, event_loop, state);
}
