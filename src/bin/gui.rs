use std::fs::File;
use ggez::*;
use neuroevolution::gui::*;

fn main() {
    let state = State {};

    let mut conf_file = File::open("gui_conf.toml").unwrap();
    let conf = conf::Conf::from_toml_file(&mut conf_file).unwrap();

    let cb = ContextBuilder::new("Neuroevolution", "Samy Haffoudhi") .default_conf(conf);
    let (ctx, event_loop) = cb.build().unwrap();

    event::run(ctx, event_loop, state);
}
