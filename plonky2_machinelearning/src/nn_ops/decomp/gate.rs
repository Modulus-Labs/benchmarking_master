use std::{fmt::Debug, marker::PhantomData};

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

use super::generator::DecompGenerator;

#[derive(Debug, Clone)]
pub struct DecompGate<
    F: RichField + Extendable<D>,
    Var: GateOuputVariant<F, D, BASE, LENGTH>,
    const D: usize,
    const BASE: u16,
    const LENGTH: usize,
> {
    pub _marker: PhantomData<(F, Var)>,
}

impl<
        F: RichField + Extendable<D>,
        Var: GateOuputVariant<F, D, BASE, LENGTH>,
        const D: usize,
        const BASE: u16,
        const LENGTH: usize,
    > DecompGate<F, Var, D, BASE, LENGTH>
{
    pub const INPUT_OFFSET: usize = 0;
    pub const BIT_SIGN_OFFSET: usize = Self::INPUT_OFFSET + 1;
    pub const DECOMP_OFFSET: usize = Self::BIT_SIGN_OFFSET + 1;
    pub const OUTPUT_OFFSET: usize = Self::DECOMP_OFFSET + LENGTH;
}

impl<
        F: RichField + Extendable<D>,
        Var: GateOuputVariant<F, D, BASE, LENGTH>,
        const D: usize,
        const BASE: u16,
        const LENGTH: usize,
    > Gate<F, D> for DecompGate<F, Var, D, BASE, LENGTH>
{
    fn id(&self) -> String {
        format!("DecompGate<D={}, Var={}>", D, Var::id())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<<F as Extendable<D>>::Extension> {
        let input = vars.local_wires[Self::INPUT_OFFSET];
        let bit_sign = vars.local_wires[Self::BIT_SIGN_OFFSET];
        let decomp_words =
            vars.local_wires[Self::DECOMP_OFFSET..Self::DECOMP_OFFSET + LENGTH].to_vec();
        let powers: Vec<_> = (0..LENGTH)
            .map(|index| {
                F::Extension::from(F::from_canonical_u16(BASE)).exp_u64(index.try_into().unwrap())
            })
            .collect();
        let output = vars.local_wires[Self::OUTPUT_OFFSET];

        let sum = decomp_words
            .clone()
            .into_iter()
            .zip(powers.clone().into_iter())
            .fold(F::Extension::ZERO, |accum, (word, power)| {
                accum + word * power
            });

        let constant_one = F::Extension::ONE;

        let output_constraint =
            Var::output_constraint(bit_sign, sum, &decomp_words, powers, output);

        let constrain_words = decomp_words.into_iter().map(|item| {
            (0..BASE).fold(F::Extension::ZERO, |accum, index| {
                accum
                    * ((F::Extension::from(F::from_canonical_u16(index.try_into().unwrap())))
                        - item)
            })
        });

        let constraints = vec![
            bit_sign * (input - sum) + ((constant_one - bit_sign) * (input + sum)),
            output_constraint,
        ];
        //ic!(output_constraint);
        constraints.into_iter().chain(constrain_words).collect()
        //vec![]
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let input = vars.local_wires[Self::INPUT_OFFSET];
        let bit_sign = vars.local_wires[Self::BIT_SIGN_OFFSET];
        let base = builder.constant_extension(F::Extension::from(F::from_canonical_u16(BASE)));
        let decomp_words =
            vars.local_wires[Self::DECOMP_OFFSET..Self::DECOMP_OFFSET + LENGTH].to_vec();
        let powers: Vec<_> = (0..LENGTH)
            .map(|index| builder.exp_u64_extension(base, index.try_into().unwrap()))
            .collect();
        let output = vars.local_wires[Self::OUTPUT_OFFSET];

        let zero = builder.zero_extension();

        let sum = decomp_words
            .clone()
            .into_iter()
            .zip(powers.clone().into_iter())
            .fold(zero, |accum, (word, power)| {
                let mul = builder.mul_extension(word, power);
                builder.add_extension(accum, mul)
            });

        let constant_one = builder.one_extension();
        let neg_bit_sign = builder.sub_extension(constant_one, bit_sign);

        let constrain_input = {
            let constrain_input_pos_sub = builder.sub_extension(input, sum);
            let constrain_input_pos = builder.mul_extension(bit_sign, constrain_input_pos_sub);

            let constrain_input_neg_add = builder.add_extension(input, sum);
            let constrain_input_neg = builder.mul_extension(neg_bit_sign, constrain_input_neg_add);
            builder.add_extension(constrain_input_pos, constrain_input_neg)
        };

        let constrain_output =
            Var::output_constraint_circuit(builder, bit_sign, sum, &decomp_words, powers, output);

        let constrain_words: Vec<_> = decomp_words.into_iter().map(|item| {
            (0..BASE).fold(builder.zero_extension(), |accum, index| {
                let index = builder.constant_extension(F::Extension::from(F::from_canonical_u16(
                    index.try_into().unwrap(),
                )));
                let sub = builder.sub_extension(index, item);
                builder.mul_extension(accum, sub)
            })
        }).collect();

        let constraints = vec![constrain_input, constrain_output];
        // constraints.into_iter().chain(constrain_words).collect()
        vec![]
    }

    fn eval_unfiltered_base_one(
        &self,
        vars_base: plonky2::plonk::vars::EvaluationVarsBase<F>,
        mut yield_constr: plonky2::gates::util::StridedConstraintConsumer<F>,
    ) {
        let input = vars_base.local_wires[Self::INPUT_OFFSET];
        let bit_sign = vars_base.local_wires[Self::BIT_SIGN_OFFSET];
        let decomp_words: Vec<_> =
            //vars_base.local_wires[Self::DECOMP_OFFSET..Self::DECOMP_OFFSET + LENGTH].to_vec();
            vars_base.local_wires.into_iter()
            .skip(Self::DECOMP_OFFSET)
            .take(LENGTH).map(|x| *x).collect();
        let powers: Vec<_> = (0..LENGTH)
            .map(|index| F::from(F::from_canonical_u16(BASE)).exp_u64(index.try_into().unwrap()))
            .collect();
        let output = vars_base.local_wires[Self::OUTPUT_OFFSET];

        let sum = decomp_words
            .clone()
            .into_iter()
            .zip(powers.clone().into_iter())
            .fold(F::ZERO, |accum, (word, power)| accum + word * power);

        let constant_one = F::ONE;

        let output_constraint =
            Var::output_constraint_base(bit_sign, sum, &decomp_words, powers, output);

        let constrain_words = decomp_words.into_iter().map(|item| {
            (0..BASE).fold(F::ZERO, |accum, index| {
                accum * ((F::from(F::from_canonical_u16(index.try_into().unwrap()))) - item)
            })
        });

        let constraints = vec![
            bit_sign * (input - sum) + ((constant_one - bit_sign) * (input + sum)),
            output_constraint,
        ];
        //ic!(output_constraint);
        // yield_constr.many(constraints);
        // yield_constr.many(constrain_words);
        //constraints.into_iter().chain(constrain_words).collect();
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<Box<dyn WitnessGenerator<F>>> {
        let gen = DecompGenerator::<F, Var, D, BASE, LENGTH> {
            row,
            _marker: PhantomData,
        };
        vec![Box::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        Self::OUTPUT_OFFSET + 1
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        BASE.into()
    }

    fn num_constraints(&self) -> usize {
        LENGTH + 2
    }
}

pub trait GateOuputVariant<
    F: RichField + Extendable<D>,
    const D: usize,
    const BASE: u16,
    const LENGTH: usize,
>: Debug + Clone + Send + Sync + 'static
{
    fn id() -> String;

    fn output_constraint(
        bit_sign: <F as Extendable<D>>::Extension,
        sum: <F as Extendable<D>>::Extension,
        decomp_words: &Vec<<F as Extendable<D>>::Extension>,
        powers: Vec<<F as Extendable<D>>::Extension>,
        output: <F as Extendable<D>>::Extension,
    ) -> <F as Extendable<D>>::Extension;

    fn output_constraint_base(
        bit_sign: F,
        sum: F,
        decomp_words: &Vec<F>,
        powers: Vec<F>,
        output: F,
    ) -> F;

    fn output_constraint_circuit(
        builder: &mut CircuitBuilder<F, D>,
        bit_sign: ExtensionTarget<D>,
        sum: ExtensionTarget<D>,
        decomp_words: &Vec<ExtensionTarget<D>>,
        powers: Vec<ExtensionTarget<D>>,
        output: ExtensionTarget<D>,
    ) -> ExtensionTarget<D>;

    fn generate_output(input: F, bit_sign: F) -> F;
}
