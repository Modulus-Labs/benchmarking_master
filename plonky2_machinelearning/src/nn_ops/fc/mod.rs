use icecream::ic;
use ndarray::Array2;
use plonky2::{
    field::extension::Extendable, hash::hash_types::RichField, iop::target::Target,
    plonk::circuit_builder::CircuitBuilder,
};
use std::marker::PhantomData;

use self::gate::FCGate;

mod gate;
mod generator;

pub trait FCCircuitBuilder<F: RichField + Extendable<D>, const D: usize> {
    fn fc_foward(
        &mut self,
        inputs: &[Target],
        weights: Array2<F>,
        biases: &[F],
        width: usize,
    ) -> Vec<Target>;
}

impl<F: RichField + Extendable<D>, const D: usize> FCCircuitBuilder<F, D> for CircuitBuilder<F, D> {
    fn fc_foward(
        &mut self,
        inputs: &[Target],
        weights: Array2<F>,
        biases: &[F],
        width: usize,
    ) -> Vec<Target> {
        let gate = FCGate::<F, D> {
            width,
            _marker: PhantomData,
        };
        //println!("fc inputs: {:?}", inputs);

        weights
            .rows()
            .into_iter()
            .zip(biases.iter())
            .map(|(weights, bias)| {
                let weights = weights.map(|x| self.constant(*x));
                let bias = self.constant(*bias);

                let row = self.add_gate(gate.clone(), vec![]);
                for (index, input) in inputs.into_iter().enumerate() {
                    self.connect(
                        *input,
                        Target::wire(row, gate.input_offset() + index),
                    );
                }
                for (index, weight) in weights.into_iter().enumerate() {
                    self.connect(
                        weight,
                        Target::wire(row, gate.weights_offset() + index),
                    );
                }
                self.connect(bias, Target::wire(row, gate.bias_offset()));

                Target::wire(row, gate.output_offset())
            })
            .collect()
    }
}
