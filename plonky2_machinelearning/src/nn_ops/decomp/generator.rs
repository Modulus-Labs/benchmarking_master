use std::marker::PhantomData;

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

use icecream::ic;

use super::gate::{DecompGate, GateOuputVariant};

#[derive(Debug)]
pub struct DecompGenerator<
    F: RichField + Extendable<D>,
    Var: GateOuputVariant<F, D, BASE, LENGTH>,
    const D: usize,
    const BASE: u16,
    const LENGTH: usize,
> {
    pub row: usize,
    pub _marker: PhantomData<(F, Var)>,
}

impl<
        F: RichField + Extendable<D>,
        Var: GateOuputVariant<F, D, BASE, LENGTH>,
        const D: usize,
        const BASE: u16,
        const LENGTH: usize,
    > SimpleGenerator<F> for DecompGenerator<F, Var, D, BASE, LENGTH>
{
    fn dependencies(&self) -> Vec<Target> {
        vec![Target::wire(
            self.row,
            DecompGate::<F, Var, D, BASE, LENGTH>::INPUT_OFFSET,
        )]
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let get_local_wire = |column| witness.get_wire(local_wire(column));

        let input = get_local_wire(DecompGate::<F, Var, D, BASE, LENGTH>::INPUT_OFFSET);

        let bit_sign = if input.to_canonical_u64() < (F::NEG_ONE / F::TWO).to_canonical_u64() {
            F::ONE
        } else {
            F::ZERO
        };

        let mut input_abs = if bit_sign == F::ONE {
            input.to_canonical_u64()
        } else {
            input.neg().to_canonical_u64()
        };

        let mut word_repr = vec![];

        loop {
            let m = input_abs % BASE as u64;
            input_abs /= BASE as u64;
            word_repr.push(F::from_canonical_u64(m));
            if input_abs == 0 {
                break;
            }
        }

        let output = Var::generate_output(input, bit_sign);

        let bit_sign_wire = local_wire(DecompGate::<F, Var, D, BASE, LENGTH>::BIT_SIGN_OFFSET);

        let word_repr_wires: Vec<_> = (0..LENGTH)
            .map(|index| local_wire(DecompGate::<F, Var, D, BASE, LENGTH>::DECOMP_OFFSET + index))
            .collect();

        let output_wire = local_wire(DecompGate::<F, Var, D, BASE, LENGTH>::OUTPUT_OFFSET);

        // ic!(bit_sign);
        // ic!(word_repr);
        // ic!(input);
        // ic!(output.to_canonical_u64());

        #[cfg(feature = "generator_test")]
        {
            let input_i64 = if input.to_canonical_u64() < (F::NEG_ONE / F::TWO).to_canonical_u64() {
                i64::try_from(input.to_canonical_u64()).unwrap()
            } else {
                -i64::try_from(input.neg().to_canonical_u64()).unwrap()
            };
            let output_real =
                if output.to_canonical_u64() < (F::NEG_ONE / F::TWO).to_canonical_u64() {
                    i64::try_from(output.to_canonical_u64()).unwrap()
                } else {
                    -i64::try_from(output.neg().to_canonical_u64()).unwrap()
                };

            let output_calc = input_i64 / i64::from(BASE.pow(4.try_into().unwrap()));
            // assert_eq!(output_calc, output_real, "calc output must equal real output; calc: {}, real: {}", output_calc, output_real);

            let input_abs_i64 = input_i64.abs();
            let word_sum_i64 = word_repr
                .iter()
                .map(|x| {
                    if x.to_canonical_u64() < (F::NEG_ONE / F::TWO).to_canonical_u64() {
                        i64::try_from(x.to_canonical_u64()).unwrap()
                    } else {
                        -i64::try_from(x.neg().to_canonical_u64()).unwrap()
                    }
                })
                .enumerate()
                .fold(0_i64, |accum, (index, item)| {
                    accum + (item * i64::try_from(BASE).unwrap().pow(index.try_into().unwrap()))
                });
            assert_eq!(
                input_abs_i64, word_sum_i64,
                "word sum must be correct, input_abs: {}, calculated word_sum: {}",
                input_abs_i64, word_sum_i64
            );

            let trunc_sum = word_repr
                .iter()
                .skip(4)
                .map(|x| {
                    if x.to_canonical_u64() < (F::NEG_ONE / F::TWO).to_canonical_u64() {
                        i64::try_from(x.to_canonical_u64()).unwrap()
                    } else {
                        -i64::try_from(x.neg().to_canonical_u64()).unwrap()
                    }
                })
                .enumerate()
                .fold(0_i64, |accum, (index, item)| {
                    accum + (item * i64::try_from(BASE).unwrap().pow(index.try_into().unwrap()))
                });

            assert_eq!(trunc_sum, output_calc.abs());

            let bit_sign_calc = if input_i64.is_negative() { 0 } else { 1 };
            assert_eq!(
                bit_sign_calc,
                bit_sign.to_canonical_u64(),
                "bit_sign must be correct, calc: {}, real: {}",
                bit_sign_calc,
                bit_sign.to_canonical_u64()
            );
        }

        out_buffer.set_wire(bit_sign_wire, bit_sign);
        out_buffer.set_wires(word_repr_wires, &word_repr);
        out_buffer.set_wire(output_wire, output);
    }
}
