use std::f64::consts::PI;
use crate::constants::ROAD_LENGTH;

const GRAVITY: f64 = -9.81;
const DELTA_T: f64 = 0.01;
const BALANCED_THRESHOLD: f64 = PI / 6.;

#[derive(Debug)]
pub struct State {
    cart_position: f64,
    cart_velocity: f64,
    pole_lengths: Vec<f64>,
    pole_angles: Vec<f64>,
    pole_velocities: Vec<f64>,
    cart_mass: f64,
    pole_masses: Vec<f64>,
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

    pub fn default() -> Self {
        Self::new(
            0.,
            0.,
            vec![1., 0.1],
            vec![0.017, 0.],
            vec![0., 0.],
            1.,
            vec![0.5, 0.05],
        )
    }

    pub fn get_cart_position(&self) -> f64 {
        self.cart_position
    }

    pub fn get_pole_angles(&self) -> Vec<f64> {
        self.pole_angles.clone()
    }

    pub fn get_pole_lengths(&self) -> Vec<f64> {
        self.pole_lengths.clone()
    }

    pub fn to_vec(&self) -> Vec<f64> {
        // TODO scaling
        let mut vec = vec![self.cart_position, self.cart_velocity];
        vec.extend(self.pole_angles.iter().cloned());
        vec.extend(self.pole_velocities.iter().cloned());
        vec
    }

    pub fn are_poles_balanced(&self) -> bool {
        self.pole_angles.iter().all(|angle| angle.abs() < BALANCED_THRESHOLD)
    }

    pub fn is_cart_out_of_bounds(&self) -> bool {
        self.cart_position.abs() > ROAD_LENGTH / 2.
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
                mass * length / 2. * velocity.powi(2) * angle.sin() + 3. / 4. * mass * GRAVITY * angle.sin() * angle.cos()
            })
            .collect::<Vec<f64>>();

        let acceleration = (force + effective_forces.iter().sum::<f64>()) / (self.cart_mass + effective_masses.iter().sum::<f64>());

        let pole_accelerations = self.pole_angles.iter()
            .zip(self.pole_lengths.iter())
            .map(|(angle, length)| { -3. / (2. * length) * (acceleration * angle.cos() + GRAVITY * angle.sin()) })
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
            vec![PI],
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
        assert!((state.pole_angles[0] - PI).abs() < TOL);
        assert!((state.pole_velocities[0] - 0.).abs() < TOL);
    }
}
