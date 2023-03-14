use std::marker::PhantomData;

use icecream::ic;
use plonky2::{
    field::{extension::Extendable, types::Field},
    gates::gate::Gate,
    hash::hash_types::RichField,
    iop::{
        ext_target::ExtensionTarget,
        generator::{SimpleGenerator, WitnessGenerator},
    },
    plonk::{
        circuit_builder::CircuitBuilder,
        vars::{EvaluationTargets, EvaluationVars},
    },
};

use crate::i64_from_felt;

use super::generator::FCGenerator;

#[derive(Debug, Clone)]
pub struct FCGate<F: RichField + Extendable<D>, const D: usize> {
    pub width: usize,
    pub _marker: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> FCGate<F, D> {
    // pub const INPUT_OFFSET: usize = 0;
    // pub const WEIGHTS_OFFSET: usize = WIDTH;
    // pub const BIAS_OFFSET: usize = Self::WEIGHTS_OFFSET;
    // pub const OUTPUT_OFFSET: usize = Self::BIAS_OFFSET + 1;

    pub fn input_offset(&self) -> usize {
        0
    }

    pub fn weights_offset(&self) -> usize {
        self.input_offset() + self.width
    }

    pub fn bias_offset(&self) -> usize {
        self.weights_offset() + self.width
    }

    pub fn output_offset(&self) -> usize {
        self.bias_offset() + 1
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D>
    for FCGate<F, D>
{
    fn id(&self) -> String {
        format!("FCGate<D={}, WIDTH={}>", D, self.width)
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<<F as Extendable<D>>::Extension> {
        let inputs: Vec<_> = (0..self.width)
            .map(|index| {
                (
                    vars.local_wires[self.input_offset() + index],
                    vars.local_wires[self.weights_offset() + index],
                )
            })
            .collect();
        let bias = vars.local_wires[self.bias_offset()];
        let output = vars.local_wires[self.output_offset()];

        let mat_mul_output = inputs
            .into_iter()
            .fold(F::Extension::ZERO, |accum, (input, weight)| {
                accum + (input * weight)
            });
        vec![output - (mat_mul_output + bias)]
        //vec![]
    }

    fn eval_unfiltered_base_one(
        &self,
        vars_base: plonky2::plonk::vars::EvaluationVarsBase<F>,
        mut yield_constr: plonky2::gates::util::StridedConstraintConsumer<F>,
    ) {
        let inputs: Vec<_> = (0..self.width)
            .map(|index| {
                (
                    vars_base.local_wires[self.input_offset()],
                    vars_base.local_wires[self.weights_offset()],
                )
            })
            .collect();
        let bias = vars_base.local_wires[self.bias_offset()];
        let output = vars_base.local_wires[self.output_offset()];

        let mat_mul_output = inputs
            .into_iter()
            .fold(F::ZERO, |accum, (input, weight)| accum + (input * weight));
        //ic!(output - (mat_mul_output + bias));
        //yield_constr.one(output - (mat_mul_output + bias));
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let inputs: Vec<_> = (0..self.width)
            .map(|index| {
                (
                    vars.local_wires[self.input_offset() + index],
                    vars.local_wires[self.weights_offset() + index],
                )
            })
            .collect();
        let bias = vars.local_wires[self.bias_offset()];
        let output = vars.local_wires[self.output_offset()];

        let mat_mul_output =
            inputs
                .into_iter()
                .fold(builder.zero_extension(), |accum, (input, weight)| {
                    let mul = builder.mul_extension(input, weight);
                    builder.add_extension(accum, mul)
                });
        let add = builder.add_extension(mat_mul_output, bias);
        // vec![builder.sub_extension(output, add)]
        vec![]
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<Box<dyn WitnessGenerator<F>>> {
        let gen = FCGenerator::<F, D> {
            row,
            gate: self.clone(),
        };
        vec![Box::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        self.output_offset() + 1
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        1
    }
}
