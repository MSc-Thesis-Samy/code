use std::f64::consts::PI;
use ggez::*;
use crate::benchmarks::LabeledPoints;
use crate::constants::UNIT_CIRCLE_STEPS;

pub struct State {}

impl State {
  pub fn new() -> Self {
    State {}
  }

  fn polar_to_canvas(&self, v: Vec<f64>) -> mint::Point2<f64> {
    let x = v[0] * v[1].cos();
    let y = v[0] * v[1].sin();
    mint::Point2{x: 400. + 250. * x,y: 300. - 250. * y}
  }
}

impl ggez::event::EventHandler<GameError> for State {
  fn update(&mut self, ctx: &mut Context) -> GameResult {
      Ok(())
  }

  fn draw(&mut self, ctx: &mut Context) -> GameResult {
    let mut canvas = graphics::Canvas::from_frame(ctx, graphics::Color::WHITE);

    // let points = vec![
    //   self.polar_to_canvas(vec![1., 0.]),
    //   self.polar_to_canvas(vec![1., PI / 2.]),
    //   self.polar_to_canvas(vec![1., PI]),
    //   self.polar_to_canvas(vec![1., 3. * PI / 2.]),
    // ];

    let points = (0..UNIT_CIRCLE_STEPS)
        .map(|i| {
            let angle = 2. * PI * i as f64 / UNIT_CIRCLE_STEPS as f64;
            (vec![1., angle], angle <= PI)
        })
        .collect::<LabeledPoints>();

    let mesh = &mut graphics::MeshBuilder::new();

    mesh.circle(
      graphics::DrawMode::stroke(2.0),
      mint::Point2{x: 400.0, y: 300.0},
      250.0,
      0.1,
      graphics::Color::BLACK,
    )?;

    for (point, label) in points {
      let point = self.polar_to_canvas(point);
      mesh.rectangle(
        graphics::DrawMode::fill(),
        graphics::Rect::new(point.x as f32 - 5., point.y as f32 - 5., 10.0, 10.0),
        if label { graphics::Color::GREEN } else { graphics::Color::RED },
      )?;
    }

    let mesh = graphics::Mesh::from_data(ctx, mesh.build());

    canvas.draw(&mesh, graphics::DrawParam::new());

    canvas.finish(ctx)?;
    Ok(())
  }
}
