use std::marker::PhantomData;

use plonky2::{
    field::{extension::Extendable, types::Field},
    hash::hash_types::RichField,
    iop::ext_target::ExtensionTarget,
    plonk::circuit_builder::CircuitBuilder,
};

use crate::nn_ops::decomp::gate::GateOuputVariant;

#[derive(Debug, Clone)]
pub struct NormalizeGateVariant<
    F: RichField + Extendable<D>,
    const D: usize,
    const BASE: u16,
    const LENGTH: usize,
    const K: usize,
> {
    _marker: PhantomData<F>,
}

impl<
        F: RichField + Extendable<D>,
        const D: usize,
        const BASE: u16,
        const LENGTH: usize,
        const K: usize,
    > GateOuputVariant<F, D, BASE, LENGTH> for NormalizeGateVariant<F, D, BASE, LENGTH, K>
{
    fn id() -> String {
        format!("Normalize<BASE={}, LENGTH={}, K={}>", BASE, LENGTH, K)
    }

    fn output_constraint(
        bit_sign: <F as Extendable<D>>::Extension,
        _sum: <F as Extendable<D>>::Extension,
        decomp_words: &Vec<<F as Extendable<D>>::Extension>,
        powers: Vec<<F as Extendable<D>>::Extension>,
        output: <F as Extendable<D>>::Extension,
    ) -> <F as Extendable<D>>::Extension {
        let trunc_sum = decomp_words
            .into_iter()
            .skip(K)
            .zip(powers)
            .fold(F::Extension::ZERO, |accum, (item, power)| {
                accum + *item * power
            });
        let constant_one = F::Extension::ONE;
        bit_sign * (output - trunc_sum) + (constant_one - bit_sign) * (output + trunc_sum)
    }

    fn output_constraint_base(
        bit_sign: F,
        sum: F,
        decomp_words: &Vec<F>,
        powers: Vec<F>,
        output: F,
    ) -> F {
        let trunc_sum = decomp_words
            .into_iter()
            .skip(K)
            .zip(powers)
            .fold(F::ZERO, |accum, (item, power)| accum + *item * power);
        let constant_one = F::ONE;
        bit_sign * (output - trunc_sum) + (constant_one - bit_sign) * (output + trunc_sum)
    }

    fn output_constraint_circuit(
        builder: &mut CircuitBuilder<F, D>,
        bit_sign: ExtensionTarget<D>,
        _sum: ExtensionTarget<D>,
        decomp_words: &Vec<ExtensionTarget<D>>,
        powers: Vec<ExtensionTarget<D>>,
        output: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let trunc_sum = decomp_words.into_iter().skip(K).zip(powers).fold(
            builder.zero_extension(),
            |accum, (item, power)| {
                let mul = builder.mul_extension(*item, power);
                builder.add_extension(accum, mul)
            },
        );

        let constant_one = builder.one_extension();
        let neg_bit_sign = builder.sub_extension(constant_one, bit_sign);

        let constrain_output_pos_sub = builder.sub_extension(output, trunc_sum);
        let constrain_output_pos = builder.mul_extension(bit_sign, constrain_output_pos_sub);

        let constrain_output_neg_add = builder.add_extension(output, trunc_sum);
        let constrain_output_neg = builder.mul_extension(neg_bit_sign, constrain_output_neg_add);
        builder.add_extension(constrain_output_pos, constrain_output_neg)
    }

    fn generate_output(input: F, bit_sign: F) -> F {
        if bit_sign == F::ONE {
            F::from_canonical_u64(
                input.to_canonical_u64() / u64::from(BASE.pow(K.try_into().unwrap())),
            )
        } else {
            F::from_canonical_u64(
                input.neg().to_canonical_u64() / u64::from(BASE.pow(K.try_into().unwrap())),
            )
            .neg()
        }
    }
}
