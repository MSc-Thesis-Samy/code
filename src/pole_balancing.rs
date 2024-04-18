const GRAVITY: f64 = 9.81;
const DELTA_T: f64 = 0.01;

#[derive(Debug)]
pub struct State {
    cart_position: f64,
    cart_velocity: f64,
    pole_angles: Vec<f64>,
    pole_lengths: Vec<f64>,
    pole_velocities: Vec<f64>,
    pole_masses: Vec<f64>,
    cart_mass: f64,
}

impl State {
    pub fn new(
        cart_position: f64,
        cart_velocity: f64,
        pole_lengths: Vec<f64>,
        pole_angles: Vec<f64>,
        pole_velocities: Vec<f64>,
        cart_mass: f64,
        pole_masses: Vec<f64>,
    ) -> Self {
        Self {
            cart_position,
            cart_velocity,
            pole_angles,
            pole_lengths,
            pole_velocities,
            pole_masses,
            cart_mass,
        }
    }

    pub fn update(&mut self, force: f64) {
        let effective_masses = std::iter::zip(self.pole_masses.iter(), self.pole_angles.iter())
            .map(|(mass, angle)| mass * (1. - 3. / 4. * angle.cos().powi(2)))
            .collect::<Vec<f64>>();

        let effective_forces = self.pole_angles.iter()
            .zip(self.pole_lengths.iter())
            .zip(self.pole_velocities.iter())
            .zip(self.pole_masses.iter())
            .map(|(((angle, length), velocity), mass)| {
                mass * length * velocity.powi(2) * angle.sin() + 3. / 4. * mass * GRAVITY * angle.sin() * angle.cos()
            })
            .collect::<Vec<f64>>();

        let acceleration = (force + effective_forces.iter().sum::<f64>()) / (self.cart_mass + effective_masses.iter().sum::<f64>());

        let pole_accelerations = self.pole_angles.iter()
            .zip(self.pole_lengths.iter())
            .map(|(angle, length)| { -3. / (4. * length) * (acceleration * angle.cos() + GRAVITY * angle.sin()) })
            .collect::<Vec<f64>>();

        self.cart_velocity += acceleration * DELTA_T;
        self.cart_position += self.cart_velocity * DELTA_T;

        self.pole_velocities = self.pole_velocities.iter()
            .zip(pole_accelerations.iter())
            .map(|(velocity, acceleration)| velocity + acceleration * DELTA_T)
            .collect::<Vec<f64>>();

        self.pole_angles = self.pole_angles.iter()
            .zip(self.pole_velocities.iter())
            .map(|(angle, velocity)| angle + velocity * DELTA_T)
            .collect::<Vec<f64>>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 5e-1;
    const TIME_STEPS: usize = 1000;

    #[test]
    fn test_pole_balancing_update_equilibrium() {
        let mut state = State::new(
            0.,
            0.,
            vec![1.],
            vec![0.],
            vec![0.],
            1.,
            vec![0.5],
        );

        let force = 0.;
        for _ in 0..TIME_STEPS {
            state.update(force);
        }

        assert!((state.cart_position - 0.).abs() < TOL);
        assert!((state.cart_velocity - 0.).abs() < TOL);
        assert!((state.pole_angles[0] - 0.).abs() < TOL);
        assert!((state.pole_velocities[0] - 0.).abs() < TOL);
    }

    #[test]
    fn test_pole_balancing_update_forward_force() {
        let mut state = State::new(
            0.,
            0.,
            vec![1.],
            vec![0.],
            vec![0.],
            1.,
            vec![0.5],
        );

        let force = 10.;
        for _ in 0..TIME_STEPS {
            state.update(force);
        }

        assert!(state.cart_position > 0.);
        assert!(state.cart_velocity > 0.);
        assert!(state.pole_angles[0] < 0.);
    }

    #[test]
    fn test_pole_balancing_update_backward_force() {
        let mut state = State::new(
            0.,
            0.,
            vec![1.],
            vec![0.],
            vec![0.],
            1.,
            vec![0.5],
        );

        let force = -10.;
        for _ in 0..TIME_STEPS {
            state.update(force);
        }

        assert!(state.cart_position < 0.);
        assert!(state.cart_velocity < 0.);
        assert!(state.pole_angles[0] > 0.);
    }

    #[test]
    fn test_pole_balancing_update_falling_pole() {
        let mut state = State::new(
            0.,
            0.,
            vec![1.],
            vec![PI / 6.],
            vec![0.],
            1.,
            vec![0.5],
        );

        let force = 0.;
        for _ in 0..TIME_STEPS {
            state.update(force);
        }

        println!("{:?}", state.pole_angles[0]);
        // assert!(state.pole_angles[0] > PI / 6.);
        assert!(state.pole_velocities[0] >= 0.);
    }
}
