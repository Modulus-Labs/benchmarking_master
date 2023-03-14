use std::marker::PhantomData;

use plonky2::{
    field::{extension::Extendable, types::Field},
    hash::hash_types::RichField,
    iop::ext_target::ExtensionTarget,
    plonk::circuit_builder::CircuitBuilder,
};

use crate::nn_ops::decomp::gate::GateOuputVariant;

#[derive(Debug, Clone)]
pub struct ReluGateVariant<
    F: RichField + Extendable<D>,
    const D: usize,
    const BASE: u16,
    const LENGTH: usize,
> {
    _marker: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize, const BASE: u16, const LENGTH: usize>
    GateOuputVariant<F, D, BASE, LENGTH> for ReluGateVariant<F, D, BASE, LENGTH>
{
    fn id() -> String {
        format!("Relu<BASE={}, LENGTH={}>", BASE, LENGTH)
    }

    fn output_constraint(
        bit_sign: <F as Extendable<D>>::Extension,
        sum: <F as Extendable<D>>::Extension,
        _decomp_words: &Vec<<F as Extendable<D>>::Extension>,
        _powers: Vec<<F as Extendable<D>>::Extension>,
        output: <F as Extendable<D>>::Extension,
    ) -> <F as Extendable<D>>::Extension {
        let constant_one = F::Extension::ONE;
        bit_sign * (output - sum) + (constant_one - bit_sign) * output
    }

    fn output_constraint_base(
        bit_sign: F,
        sum: F,
        decomp_words: &Vec<F>,
        powers: Vec<F>,
        output: F,
    ) -> F {
        let constant_one = F::ONE;
        bit_sign * (output - sum) + (constant_one - bit_sign) * output
    }

    fn output_constraint_circuit(
        builder: &mut CircuitBuilder<F, D>,
        bit_sign: ExtensionTarget<D>,
        sum: ExtensionTarget<D>,
        _decomp_words: &Vec<ExtensionTarget<D>>,
        _powers: Vec<ExtensionTarget<D>>,
        output: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let constant_one = builder.one_extension();
        let neg_bit_sign = builder.sub_extension(constant_one, bit_sign);

        let constrain_output_pos_sub = builder.sub_extension(output, sum);
        let constrain_output_pos = builder.mul_extension(bit_sign, constrain_output_pos_sub);

        let constrain_output_neg = builder.mul_extension(neg_bit_sign, output);
        builder.add_extension(constrain_output_pos, constrain_output_neg)
    }

    fn generate_output(input: F, bit_sign: F) -> F {
        if bit_sign == F::ONE {
            input
        } else {
            F::ZERO
        }
    }
}
