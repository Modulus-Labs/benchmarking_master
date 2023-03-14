use anyhow::Result;
use icecream::ic;
use ndarray::array;
use log::Level;
use plonky2::{
    field::types::{Field, PrimeField64},
    iop::{
        target::Target,
        witness::{PartialWitness, Witness},
    },
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::CircuitConfig,
        config::{GenericConfig, PoseidonGoldilocksConfig},
        prover::prove
    },
    util::timing::TimingTree,
};
use plonky2_machinelearning::{
    felt_from_i64,
    nn_ops::{
        decomp::{variants::relu_normalize::ReluNormalizeGateVariant, DecompCircuitBuilder},
        fc::FCCircuitBuilder,
    },
};

fn main() -> Result<()> {
    simple_logger::SimpleLogger::new().init().unwrap();
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    const HEIGHT: usize = 4;
    const WIDTH: usize = 3;

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let weights = (array![
        [4_096, -8_192, 4_096],
        [-8_192, 4_096, 4_096],
        [-20_480, 4_096, 4_096],
        [4_096, 4_096, 4_096]
    ])
    .map(|x| felt_from_i64::<F, D>(*x));
    let biases: [F; HEIGHT] = (vec![-16_777_216, 16_777_216, 33_554_432, 16_777_216])
        .into_iter()
        .map(|x| felt_from_i64::<_, D>(x))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let inputs: [Target; WIDTH] = (0..3)
        .map(|_| builder.add_virtual_public_input())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let weights_2 = (array![
        [4_096, -8_192, 4_096, 4_096],
        [-8_192, 4_096, 4_096, 4_096],
        [-20_480, 4_096, 4_096, 4_096],
        [4_096, 4_096, 4_096, 4_096]
    ])
    .map(|x| felt_from_i64::<F, D>(*x));

    let weights_3 = (array![
        [4_096, -8_192, 4_096, 4_096],
        [-8_192, 4_096, 4_096, 4_096],
        [-20_480, 4_096, 4_096, 4_096],
    ])
    .map(|x| felt_from_i64::<_, D>(*x));

    let outputs = builder.fc_foward::<3, 4>(&inputs, weights, biases.as_slice());

    let outputs =
        builder.decomp_eltwise::<ReluNormalizeGateVariant<F, D, 8, 15, 4>, 8, 15>(outputs);

    let outputs = builder.fc_foward::<4, 3>(&outputs, weights_3, biases.as_slice());

    let outputs =
        builder.decomp_eltwise::<ReluNormalizeGateVariant<F, D, 8, 15, 4>, 8, 15>(outputs);

    builder.register_public_inputs(&outputs);

    let mut pw = PartialWitness::new();
    pw.set_target_arr(
        inputs.try_into().unwrap(),
        [
            F::from_canonical_u16(8_192),
            F::from_canonical_u16(12_288),
            F::from_canonical_u16(16_384),
            //F::from_canonical_u16(16_384),
        ],
    );
    let data = builder.build::<C>();
    let mut timing = TimingTree::new("prove", Level::Info);
    let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
    timing.print();
    //let proof = data.prove(pw)?;
    println!("proof size is: {:?}", proof.to_bytes()?.len());

    println!(
        "public_inputs are {:?}",
        proof
            .public_inputs
            .iter()
            .map(|x| x.to_canonical_u64())
            .collect::<Vec<_>>()
    );

    data.verify(proof)?;
    println!("done!");
    Ok(())
}
