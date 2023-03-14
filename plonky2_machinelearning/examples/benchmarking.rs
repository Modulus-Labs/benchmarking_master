#![recursion_limit = "512"]
use std::ops::Neg;

use anyhow::{Result, Error};
use log::{Level, LevelFilter};

use ndarray::{array, stack, Array1, Array2, ArrayView, Axis};
use plonky2::{
    field::{
        extension::Extendable,
        types::{Field, PrimeField64},
    },
    fri::{reduction_strategies::FriReductionStrategy, FriConfig},
    hash::hash_types::RichField,
    iop::{
        target::Target,
        witness::{PartialWitness, Witness},
    },
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::{CircuitConfig, VerifierCircuitTarget},
        config::{GenericConfig, KeccakGoldilocksConfig, PoseidonGoldilocksConfig},
        prover::prove,
    },
    util::timing::TimingTree, gates::noop::NoopGate
};
use plonky2_machinelearning::{
    felt_from_i64, i64_from_felt,
    nn_ops::{
        decomp::{
            variants::{normalize::NormalizeGateVariant, relu_normalize::ReluNormalizeGateVariant},
            DecompCircuitBuilder,
        },
        fc::FCCircuitBuilder,
    },
};

type NetworkArch = &'static [(usize, usize, bool)];

use std::time::Instant;

use icecream::ic;

const DECOMP_LENGTH: usize = 10;
const DECOMP_BASE: u16 = 8;
const NORM_K: usize = 4;

// macro_rules! add_layers {
//     ($builder:expr, $layers:expr, $inputs:expr, $layer_count:expr; ($width:literal, $height:literal, $relu:expr)) => {{
//         let rows: Vec<_> = $layers[$layer_count].weights.chunks($width).map(|row| {
//             ArrayView::from(row)
//         }).collect();
//         let weights = stack(Axis(0), rows.as_slice())?;
//         let mat_output = FCCircuitBuilder::fc_foward::<$width, $height>($builder, &$inputs, weights, $layers[$layer_count].biases.as_slice());
//         if $relu {DecompCircuitBuilder::decomp_eltwise::<ReluNormalizeGateVariant<F, D, DECOMP_BASE, DECOMP_LENGTH, NORM_K>, DECOMP_BASE, DECOMP_LENGTH>($builder, mat_output)}
//         else {DecompCircuitBuilder::decomp_eltwise::<NormalizeGateVariant<F, D, DECOMP_BASE, DECOMP_LENGTH, NORM_K>, DECOMP_BASE, DECOMP_LENGTH>($builder, mat_output)}
//     }};

//     ($builder:expr, $layers:expr, $inputs:expr, $layer_count:expr; ($width:literal, $height:literal, $relu:expr), $(($width_extra:literal, $height_extra:literal, $relu_extra:expr)),+) => {{
//         let rows: Vec<_> = $layers[$layer_count].weights.chunks($width).map(|row| {
//             ArrayView::from(row)
//         }).collect();
//         let weights = stack(Axis(0), rows.as_slice())?;
//         let mat_output = FCCircuitBuilder::fc_foward::<$width, $height>($builder, &$inputs, weights, $layers[$layer_count].biases.as_slice());
//         let outputs = if $relu {DecompCircuitBuilder::decomp_eltwise::<ReluNormalizeGateVariant<F, D, DECOMP_BASE, DECOMP_LENGTH, NORM_K>, DECOMP_BASE, DECOMP_LENGTH>($builder, mat_output)}
//         else {DecompCircuitBuilder::decomp_eltwise::<NormalizeGateVariant<F, D, DECOMP_BASE, DECOMP_LENGTH, NORM_K>, DECOMP_BASE, DECOMP_LENGTH>($builder, mat_output)};
//         add_layers!($builder, $layers, outputs, $layer_count+1; $(($width_extra, $height_extra, $relu_extra)),+)
//     }};

//     ($builder:expr, $layers:expr, $inputs:expr; $(($width_extra:literal, $height_extra:literal, $relu_extra:expr)),+) => {
//         add_layers!($builder, $layers, $inputs, 0; $(($width_extra, $height_extra, $relu_extra)),+)
//     };
// }

// macro_rules! bench_nn {
//     ($net_name:expr, $file_path:expr, $largest_width:literal, $(($width_extra:literal, $height_extra:literal, $relu_extra:expr)),+) => {{
//         println!("Benches for NN: {:?}", $net_name);
//         println!("-------------");
//         #[cfg(feature = "dhat-heap")]
//         let _profiler = dhat::Profiler::builder().testing().build();

//         const LARGEST_WIDTH: usize = $largest_width;

//         let (inputs, layers, _) = get_inputs::<F, D>(
//             $file_path
//         )?;
        
//         let config = CircuitConfig {
//             num_wires: LARGEST_WIDTH * 2 + 100,
//             num_routed_wires: LARGEST_WIDTH * 2 + 100,
//             num_constants: 2,
//             use_base_arithmetic_gate: true,
//             security_bits: 100,
//             num_challenges: 2,
//             zero_knowledge: false,
//             max_quotient_degree_factor: 8,
//             fri_config: FriConfig {
//                 rate_bits: 3,
//                 cap_height: 4,
//                 proof_of_work_bits: 16,
//                 reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
//                 num_query_rounds: 28,
//             },
//         };
    
//         //let config = CircuitConfig::wide_ecc_config();
    
//         let mut builder = CircuitBuilder::<F, D>::new(config);
    
//         let input_targets: Vec<_> = (0..inputs.len())
//             .map(|_| builder.add_virtual_public_input())
//             .collect();
//         //builder.fc_foward::<3, 4>(&input_targets, weights, layers[0].biases.as_slice());
//         let outputs =
//             add_layers!(&mut builder, layers, input_targets; $(($width_extra, $height_extra, $relu_extra)),+);
    
//         //ic!(outputs);
    
//         builder.register_public_inputs(&outputs);

//         let data = builder.build::<C>();
    
//         let mut pw = PartialWitness::new();
    
//         for (target, input) in input_targets.iter().zip(inputs.iter()) {
//             pw.set_target(*target, *input);
//         }
//         let mut timing = TimingTree::new(&format!("prove {}", $net_name), Level::Info);
    
//         let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
//         timing.print();
//         println!("proof size is: {:?}", proof.to_bytes()?.len());
    
//         // println!(
//         //     "outputs are {:?}",
//         //     proof
//         //         .public_inputs
//         //         .iter()
//         //         .map(
//         //             |x| i64_from_felt::<F, D>(*x)
//         //         )
//         //         .skip(inputs.len())
//         //         .collect::<Vec<_>>()
//         // );
    
//         let now = Instant::now();
    
//         data.verify(proof)?;
//         println!("done; verif took {:?}", now.elapsed().as_secs_f64());
//         #[cfg(feature = "dhat-heap")]
//         {
//             let stats = dhat::HeapStats::get();
//             println!("mem used: {:?}", stats.max_bytes);
//         }    
//         println!("-------------");
    
//         Ok::<(), Error>(())    
//     }};
// }

pub struct LayerParams<F: RichField + Extendable<D>, const D: usize> {
    pub weights: Vec<F>,
    pub biases: Vec<F>,
}

type C = PoseidonGoldilocksConfig;
const D: usize = 2;
type F = <C as GenericConfig<D>>::F;

fn bench_nn(net_name: &str, file_path: &str, largest_width: usize, network: NetworkArch) -> Result<(), Error> {
    println!("Benches for NN: {:?}", net_name);
    println!("-------------");
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::builder().testing().build();

    let (inputs, layers, _) = get_inputs::<F, D>(
        file_path
    )?;
    
    let config = CircuitConfig {
        num_wires: largest_width * 2 + 10,
        num_routed_wires: largest_width * 2 + 10,
        num_constants: 2,
        use_base_arithmetic_gate: true,
        security_bits: 100,
        num_challenges: 2,
        zero_knowledge: false,
        max_quotient_degree_factor: 8,
        fri_config: FriConfig {
            rate_bits: 3,
            cap_height: 4,
            proof_of_work_bits: 16,
            reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
            num_query_rounds: 28,
        },
    };

    // let config = CircuitConfig::standard_recursion_config();

    let mut builder = CircuitBuilder::<F, D>::new(config);

    let input_targets: Vec<_> = (0..inputs.len())
        .map(|_| builder.add_virtual_public_input())
        .collect();

    let mut inter_targets = input_targets.clone();
    //builder.fc_foward::<3, 4>(&input_targets, weights, layers[0].biases.as_slice());
    // let outputs =
    //     add_layers!(&mut builder, layers, input_targets; $(($width_extra, $height_extra, $relu_extra)),+);
    for ((width, height, relu), layer) in network.iter().zip(layers.iter()) {
        let rows: Vec<_> = layer.weights.chunks(*width).map(|row| {
            ArrayView::from(row)
        }).collect();
        let weights = stack(Axis(0), rows.as_slice())?;
        let mat_output = FCCircuitBuilder::fc_foward(&mut builder, inter_targets.as_slice(), weights, layer.biases.as_slice(), *width);
        inter_targets = if *relu {DecompCircuitBuilder::decomp_eltwise::<ReluNormalizeGateVariant<F, D, DECOMP_BASE, DECOMP_LENGTH, NORM_K>, DECOMP_BASE, DECOMP_LENGTH>(&mut builder, mat_output)}
        else {DecompCircuitBuilder::decomp_eltwise::<NormalizeGateVariant<F, D, DECOMP_BASE, DECOMP_LENGTH, NORM_K>, DECOMP_BASE, DECOMP_LENGTH>(&mut builder, mat_output)};
    }

    //ic!(outputs);

    builder.register_public_inputs(&inter_targets);

    let mut data = builder.build::<C>();

    let mut pw = PartialWitness::new();

    for (target, input) in input_targets.iter().zip(inputs.iter()) {
        pw.set_target(*target, *input);
    }
    let mut timing = TimingTree::new(&format!("prove {}", net_name), Level::Info);

    let mut proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
    timing.print();
    println!("proof size is: {:?}", proof.to_bytes()?.len());

    // println!(
    //     "outputs are {:?}",
    //     proof
    //         .public_inputs
    //         .iter()
    //         .map(
    //             |x| i64_from_felt::<F, D>(*x)
    //         )
    //         .skip(inputs.len())
    //         .collect::<Vec<_>>()
    // );

    //let now = Instant::now();

    //data.verify(proof)?;
    //println!("done; verif took {:?}", now.elapsed().as_secs_f64());
    let mut recursion_count = 1;
    let mut prev_byte_count = 0;
    let mut cur_byte_count = 0;
    let mut first = true;

    while (proof.to_bytes()?.len() > 200000 && (prev_byte_count - cur_byte_count) > 10000) || first {
        prev_byte_count = cur_byte_count;
        first = false;

        let (inner_proof, inner_vd, inner_cd) = (&proof, &data.verifier_only, &data.common);

        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let mut pw_recurse = PartialWitness::new();
        let pt = builder.add_virtual_proof_with_pis(inner_cd);
        pw_recurse.set_proof_with_pis_target(&pt, inner_proof);

        let inner_data = VerifierCircuitTarget {
            constants_sigmas_cap: builder.add_virtual_cap(inner_cd.config.fri_config.cap_height),
            circuit_digest: builder.add_virtual_hash(),
        };
        pw_recurse.set_verifier_data_target(&inner_data, inner_vd);

        builder.verify_proof(pt, &inner_data, inner_cd);
        builder.print_gate_counts(0);


        data = builder.build::<C>();
        println!("starting to prove recursion");
        let mut timing = TimingTree::new(&format!("prove {} recursion {}", net_name, recursion_count), Level::Info);
        proof = prove(&data.prover_only, &data.common, pw_recurse, &mut timing)?;
        timing.print();
        cur_byte_count = proof.to_bytes()?.len();
        println!("proof size after {} recursions is: {:?}", recursion_count, cur_byte_count);
        recursion_count += 1;
    }

    data.verify(proof)?;

    #[cfg(feature = "dhat-heap")]
    {
        let stats = dhat::HeapStats::get();
        println!("mem used: {:?}", stats.max_bytes);
    }    
    println!("-------------");

    Ok::<(), Error>(())    

}

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> Result<()> {
    simple_logger::SimpleLogger::new().with_level(LevelFilter::Info).init().unwrap();


    //plonky2_deep_small_bench
    {
        const NETWORK3: NetworkArch = &[(32, 100, true), (100, 200, true), (200, 100, false)];
        bench_nn("FC_3_layers", "plonky2_deep_small_bench/FC_3_layers.json", 200, NETWORK3)?;
        const NETWORK0: NetworkArch = &[(32, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 100, false), ];
        bench_nn("FC_10_layers", "plonky2_deep_small_bench/FC_10_layers.json", 200, NETWORK0)?;
        const NETWORK1: NetworkArch = &[(32, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, false), ];
        bench_nn("FC_15_layers", "plonky2_deep_small_bench/FC_15_layers.json", 200, NETWORK1)?;
        const NETWORK2: NetworkArch = &[(32, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 100, false), ];
        bench_nn("FC_20_layers", "plonky2_deep_small_bench/FC_20_layers.json", 200, NETWORK2)?;
        const NETWORK4: NetworkArch = &[(32, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 100, false), ];
        bench_nn("FC_30_layers", "plonky2_deep_small_bench/FC_30_layers.json", 200, NETWORK4)?;
        const NETWORK5: NetworkArch = &[(32, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 100, false), ];
        bench_nn("FC_40_layers", "plonky2_deep_small_bench/FC_40_layers.json", 200, NETWORK5)?;
        const NETWORK6: NetworkArch = &[(32, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, false), ];
        bench_nn("FC_5_layers", "plonky2_deep_small_bench/FC_5_layers.json", 200, NETWORK6)?;

    }
    //plonky2_deep_bench
    {
        const NETWORK0: NetworkArch = &[(784, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 1000, false), ];
        bench_nn("FC_100_layers", "plonky2_deep_bench/FC_100_layers.json", 784, NETWORK0)?;
        const NETWORK1: NetworkArch = &[(784, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 1000, false), ];
        bench_nn("FC_150_layers", "plonky2_deep_bench/FC_150_layers.json", 784, NETWORK1)?;
        const NETWORK2: NetworkArch = &[(784, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 1000, false), ];
        bench_nn("FC_200_layers", "plonky2_deep_bench/FC_200_layers.json", 784, NETWORK2)?;
        const NETWORK3: NetworkArch = &[(784, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 1000, false), ];
        bench_nn("FC_350_layers", "plonky2_deep_bench/FC_350_layers.json", 784, NETWORK3)?;
        const NETWORK4: NetworkArch = &[(784, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 1000, false), ];
        bench_nn("FC_50_layers", "plonky2_deep_bench/FC_50_layers.json", 784, NETWORK4)?;
        const NETWORK5: NetworkArch = &[(784, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 1000, false), ];
        bench_nn("FC_500_layers", "plonky2_deep_bench/FC_500_layers.json", 784, NETWORK5)?;
        const NETWORK6: NetworkArch = &[(784, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 100, true), (100, 200, true), (200, 1000, false), ];
        bench_nn("FC_75_layers", "plonky2_deep_bench/FC_75_layers.json", 784, NETWORK6)?;

    }
    //plonky2_num_params_bench 
    {
        const NETWORK0: NetworkArch = &[(784, 95, true), (95, 104, true), (104, 114, true), (114, 126, true), (126, 139, true), (139, 152, true), (152, 168, true), (168, 185, true), (185, 203, true), (203, 224, true), (224, 246, true), (246, 271, true), (271, 298, true), (298, 327, true), (327, 360, true), (360, 396, true), (396, 436, true), (436, 480, true), (480, 528, true), (528, 581, true), (581, 639, true), (639, 703, true), (703, 773, true), (773, 850, true), (850, 935, true), (935, 1029, true), (1029, 1132, true), (1132, 1245, true), (1245, 1369, true), (1369, 1000, false), ];
        bench_nn("FC_11216646_params", "plonky2_num_params_bench/FC_11216646_params.json", 1369, NETWORK0)?;
        const NETWORK1: NetworkArch = &[(784, 27, true), (27, 29, true), (29, 32, true), (32, 35, true), (35, 39, true), (39, 43, true), (43, 47, true), (47, 52, true), (52, 57, true), (57, 63, true), (63, 70, true), (70, 77, true), (77, 84, true), (84, 93, true), (93, 102, true), (102, 112, true), (112, 124, true), (124, 136, true), (136, 150, true), (150, 165, true), (165, 181, true), (181, 199, true), (199, 219, true), (219, 241, true), (241, 265, true), (265, 292, true), (292, 321, true), (321, 353, true), (353, 389, true), (389, 1000, false), ];
        bench_nn("FC_1195804_params", "plonky2_num_params_bench/FC_1195804_params.json", 784, NETWORK1)?;
        const NETWORK2: NetworkArch = &[(784, 110, true), (110, 121, true), (121, 133, true), (133, 146, true), (146, 161, true), (161, 177, true), (177, 194, true), (194, 214, true), (214, 235, true), (235, 259, true), (259, 285, true), (285, 313, true), (313, 345, true), (345, 379, true), (379, 417, true), (417, 459, true), (459, 505, true), (505, 555, true), (555, 611, true), (611, 672, true), (672, 740, true), (740, 814, true), (814, 895, true), (895, 984, true), (984, 1083, true), (1083, 1191, true), (1191, 1310, true), (1310, 1442, true), (1442, 1586, true), (1586, 1000, false), ];
        bench_nn("FC_14773663_params", "plonky2_num_params_bench/FC_14773663_params.json", 1586, NETWORK2)?;
        const NETWORK3: NetworkArch = &[(784, 123, true), (123, 135, true), (135, 148, true), (148, 163, true), (163, 180, true), (180, 198, true), (198, 217, true), (217, 239, true), (239, 263, true), (263, 290, true), (290, 319, true), (319, 350, true), (350, 386, true), (386, 424, true), (424, 467, true), (467, 513, true), (513, 565, true), (565, 621, true), (621, 683, true), (683, 752, true), (752, 827, true), (827, 910, true), (910, 1001, true), (1001, 1101, true), (1101, 1211, true), (1211, 1332, true), (1332, 1465, true), (1465, 1612, true), (1612, 1773, true), (1773, 1000, false), ];
        bench_nn("FC_18252933_params", "plonky2_num_params_bench/FC_18252933_params.json", 1773, NETWORK3)?;
        const NETWORK4: NetworkArch = &[(784, 39, true), (39, 42, true), (42, 47, true), (47, 51, true), (51, 57, true), (57, 62, true), (62, 69, true), (69, 75, true), (75, 83, true), (83, 91, true), (91, 101, true), (101, 111, true), (111, 122, true), (122, 134, true), (134, 148, true), (148, 162, true), (162, 179, true), (179, 197, true), (197, 216, true), (216, 238, true), (238, 262, true), (262, 288, true), (288, 317, true), (317, 349, true), (349, 384, true), (384, 422, true), (422, 464, true), (464, 511, true), (511, 562, true), (562, 1000, false), ];
        bench_nn("FC_2236477_params", "plonky2_num_params_bench/FC_2236477_params.json", 784, NETWORK4)?;
        const NETWORK5: NetworkArch = &[(784, 55, true), (55, 60, true), (60, 66, true), (66, 73, true), (73, 80, true), (80, 88, true), (88, 97, true), (97, 107, true), (107, 117, true), (117, 129, true), (129, 142, true), (142, 156, true), (156, 172, true), (172, 189, true), (189, 208, true), (208, 229, true), (229, 252, true), (252, 277, true), (277, 305, true), (305, 336, true), (336, 370, true), (370, 407, true), (407, 447, true), (447, 492, true), (492, 541, true), (541, 595, true), (595, 655, true), (655, 721, true), (721, 793, true), (793, 1000, false), ];
        bench_nn("FC_4107399_params", "plonky2_num_params_bench/FC_4107399_params.json", 793, NETWORK5)?;
        const NETWORK6: NetworkArch = &[(784, 16, true), (16, 17, true), (17, 19, true), (19, 21, true), (21, 23, true), (23, 25, true), (25, 28, true), (28, 31, true), (31, 34, true), (34, 37, true), (37, 41, true), (41, 45, true), (45, 50, true), (50, 55, true), (55, 60, true), (60, 66, true), (66, 73, true), (73, 80, true), (80, 88, true), (88, 97, true), (97, 107, true), (107, 118, true), (118, 130, true), (130, 143, true), (143, 157, true), (157, 173, true), (173, 190, true), (190, 209, true), (209, 230, true), (230, 1000, false), ];
        bench_nn("FC_517529_params", "plonky2_num_params_bench/FC_517529_params.json", 784, NETWORK6)?;
        const NETWORK7: NetworkArch = &[(784, 19, true), (19, 20, true), (20, 22, true), (22, 25, true), (25, 27, true), (27, 30, true), (30, 33, true), (33, 37, true), (37, 40, true), (40, 44, true), (44, 49, true), (49, 54, true), (54, 59, true), (59, 65, true), (65, 72, true), (72, 79, true), (79, 87, true), (87, 96, true), (96, 105, true), (105, 116, true), (116, 127, true), (127, 140, true), (140, 154, true), (154, 170, true), (170, 187, true), (187, 205, true), (205, 226, true), (226, 249, true), (249, 273, true), (273, 1000, false), ];
        bench_nn("FC_676836_params", "plonky2_num_params_bench/FC_676836_params.json", 784, NETWORK7)?;
        const NETWORK8: NetworkArch = &[(784, 78, true), (78, 85, true), (85, 94, true), (94, 103, true), (103, 114, true), (114, 125, true), (125, 138, true), (138, 151, true), (151, 167, true), (167, 183, true), (183, 202, true), (202, 222, true), (222, 244, true), (244, 269, true), (269, 296, true), (296, 325, true), (325, 358, true), (358, 394, true), (394, 433, true), (433, 477, true), (477, 524, true), (524, 577, true), (577, 634, true), (634, 698, true), (698, 768, true), (768, 845, true), (845, 929, true), (929, 1022, true), (1022, 1124, true), (1124, 1000, false), ];
        bench_nn("FC_7770136_params", "plonky2_num_params_bench/FC_7770136_params.json", 1124, NETWORK8)?;
    }
    //plonky2_num_params_small_bench
    {
        const NETWORK0: NetworkArch = &[(32, 32, true), (32, 35, true), (35, 38, true), (38, 42, true), (42, 46, true), (46, 51, true), (51, 56, true), (56, 62, true), (62, 68, true), (68, 75, true), (75, 82, true), (82, 91, true), (91, 100, true), (100, 110, true), (110, 121, true), (121, 133, true), (133, 147, true), (147, 161, true), (161, 177, true), (177, 100, false), ];
        bench_nn("FC_177522_params", "plonky2_num_params_small_bench/FC_177522_params.json", 177, NETWORK0)?;
        const NETWORK1: NetworkArch = &[(32, 16, true), (16, 17, true), (17, 19, true), (19, 21, true), (21, 23, true), (23, 25, true), (25, 28, true), (28, 31, true), (31, 34, true), (34, 37, true), (37, 41, true), (41, 45, true), (45, 50, true), (50, 55, true), (55, 60, true), (60, 66, true), (66, 73, true), (73, 80, true), (80, 88, true), (88, 100, false), ];
        bench_nn("FC_48564_params", "plonky2_num_params_small_bench/FC_48564_params.json", 88, NETWORK1)?;
        const NETWORK2: NetworkArch = &[(32, 22, true), (22, 24, true), (24, 26, true), (26, 29, true), (29, 32, true), (32, 35, true), (35, 38, true), (38, 42, true), (42, 47, true), (47, 51, true), (51, 57, true), (57, 62, true), (62, 69, true), (69, 75, true), (75, 83, true), (83, 91, true), (91, 101, true), (101, 111, true), (111, 122, true), (122, 100, false), ];
        bench_nn("FC_87771_params", "plonky2_num_params_small_bench/FC_87771_params.json", 122, NETWORK2)?;
    }

    Ok(())
}

fn get_inputs<F: RichField + Extendable<D>, const D: usize>(
    file_name: &str,
) -> Result<(Vec<F>, Vec<LayerParams<F, D>>, Vec<F>)> {
    //const PREFIX: &str = "/home/aweso/plonky2/plonky2_machinelearning/bench_objects/";
    const PREFIX: &str = "/home/ubuntu/plonky2_bench/bench_objects/";
    let inputs_raw = std::fs::read_to_string(PREFIX.to_owned()+file_name)?;
    let inputs = json::parse(&inputs_raw)?;
    let input: Vec<_> = inputs["input"]
        .members()
        .map(|x| felt_from_i64(x.as_i64().unwrap()))
        .collect();
    let layers: Vec<LayerParams<F, D>> = inputs["layers"]
        .members()
        .map(|layer| LayerParams {
            weights: layer["weight"]
                .members()
                .map(|x| felt_from_i64(x.as_i64().unwrap()))
                .collect(),
            biases: layer["bias"]
                .members()
                .map(|x| felt_from_i64(x.as_i64().unwrap()))
                .collect(),
        })
        .collect();

    let output: Vec<_> = inputs["output"]
        .members()
        .map(|x| felt_from_i64(x.as_i64().unwrap()))
        .collect();

    Ok((input, layers, output))
}
