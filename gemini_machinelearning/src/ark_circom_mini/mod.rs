use ark_ec::pairing::Pairing as PairingEngine;

pub mod r1cs_reader;
pub use r1cs_reader::{R1CSFile, R1CS};
pub mod witness;

mod circuit;
pub use circuit::CircomCircuit;

mod builder;
pub use builder::{CircomBuilder, CircomConfig};

// mod qap;
// pub use qap::CircomReduction;

pub type Constraints<E> = (ConstraintVec<E>, ConstraintVec<E>, ConstraintVec<E>);
pub type ConstraintVec<E> = Vec<(usize, <E as PairingEngine>::ScalarField)>;
