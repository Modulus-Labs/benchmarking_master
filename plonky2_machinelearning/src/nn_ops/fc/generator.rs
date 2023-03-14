use icecream::ic;
use plonky2::{
    field::extension::Extendable,
    hash::hash_types::RichField,
    iop::{
        generator::{GeneratedValues, SimpleGenerator},
        target::Target,
        wire::Wire,
        witness::{PartitionWitness, Witness},
    },
};

use super::gate::FCGate;

#[derive(Debug)]
pub struct FCGenerator<F: RichField + Extendable<D>, const D: usize> {
    pub row: usize,
    pub gate: FCGate<F, D>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F>
    for FCGenerator<F, D>
{
    fn dependencies(&self) -> Vec<Target> {
        let local_target = |column| Target::wire(self.row, column);
        let width = self.gate.width;

        let inputs: Vec<_> = (0..width)
            .map(|index| local_target(self.gate.input_offset() + index))
            .collect();
        let weights: Vec<_> = (0..width)
            .map(|index| local_target(self.gate.weights_offset() + index))
            .collect();
        let bias = local_target(self.gate.bias_offset());

        let mut deps: Vec<_> = inputs.into_iter().chain(weights.into_iter()).collect();
        deps.push(bias);
        deps
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };
        let width = self.gate.width;

        let get_local_wire = |column| witness.get_wire(local_wire(column));
        let inputs: Vec<_> = (0..width)
            .map(|index| {
                (
                    get_local_wire(self.gate.input_offset() + index),
                    get_local_wire(self.gate.weights_offset() + index),
                )
            })
            .collect();
        let bias = get_local_wire(self.gate.bias_offset());

        let output = inputs
            .into_iter()
            .fold(F::ZERO, |accum, (input, weight)| accum + (input * weight))
            + bias;

        let output_wire = local_wire(self.gate.output_offset());

        out_buffer.set_wire(output_wire, output)
    }
}
