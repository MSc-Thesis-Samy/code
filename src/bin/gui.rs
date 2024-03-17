use std::f64::consts::PI;
use std::fs::File;
use ggez::*;
use neuroevolution::gui::*;
use neuroevolution::network::Network;
use neuroevolution::neuroevolution_algorithm::*;
use neuroevolution::constants::*;
use neuroevolution::benchmarks::*;

fn main() {
    let network = Network::new(1, 2);
    let mut alg = Algorithm::ContinuousOneplusoneNA(network);
    alg.optimize(half, N_ITERATIONS);

    // let points = (0..UNIT_CIRCLE_STEPS)
    //     .map(|i| {
    //         let angle = 2. * PI * i as f64 / UNIT_CIRCLE_STEPS as f64;
    //         (vec![1., angle], angle <= PI)
    //     })
    //     .collect::<LabeledPoints>();

    let points = LabeledPoints::new();

    let state = State::new(alg, points);

    let mut conf_file = File::open("gui_conf.toml").unwrap();
    let conf = conf::Conf::from_toml_file(&mut conf_file).unwrap();

    let cb = ContextBuilder::new("Neuroevolution", "Samy Haffoudhi") .default_conf(conf);
    let (ctx, event_loop) = cb.build().unwrap();

    event::run(ctx, event_loop, state);
}
