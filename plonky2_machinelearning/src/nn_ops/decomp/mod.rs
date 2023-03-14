mod gate;
mod generator;
pub mod variants;

use plonky2::{
    field::extension::Extendable, hash::hash_types::RichField, iop::target::Target,
    plonk::circuit_builder::CircuitBuilder,
};
use std::marker::PhantomData;

use self::gate::{DecompGate, GateOuputVariant};

pub trait DecompCircuitBuilder<F: RichField + Extendable<D>, const D: usize> {
    fn decomp_eltwise<
        Var: GateOuputVariant<F, D, BASE, LENGTH>,
        const BASE: u16,
        const LENGTH: usize,
    >(
        &mut self,
        inputs: Vec<Target>,
    ) -> Vec<Target>;
}

impl<F: RichField + Extendable<D>, const D: usize> DecompCircuitBuilder<F, D>
    for CircuitBuilder<F, D>
{
    fn decomp_eltwise<
        Var: GateOuputVariant<F, D, BASE, LENGTH>,
        const BASE: u16,
        const LENGTH: usize,
    >(
        &mut self,
        inputs: Vec<Target>,
    ) -> Vec<Target> {
        let gate = DecompGate::<F, Var, D, BASE, LENGTH> {
            _marker: PhantomData,
        };

        //println!("decomp inputs: {:?}", inputs);

        inputs
            .into_iter()
            .map(|input| {
                let row = self.add_gate(gate.clone(), vec![]);
                self.connect(
                    input,
                    Target::wire(row, <DecompGate<F, Var, D, BASE, LENGTH>>::INPUT_OFFSET),
                );
                Target::wire(row, <DecompGate<F, Var, D, BASE, LENGTH>>::OUTPUT_OFFSET)
            })
            .collect()
    }
}
