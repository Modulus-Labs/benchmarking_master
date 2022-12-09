use ark_gemini::circuit::generate_relation;
use r1cs_machinelearning::ark_circom_mini::{CircomBuilder, CircomConfig};

use ark_ec::bls12::Bls12;
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_std::test_rng;

use std::time::Instant;

type Proof = ark_gemini::snark::Proof<ark_bls12_381::Bls12_381>;
use ark_bls12_381::Bls12_381;

#[derive(Default, Clone, Debug)]
pub struct LayerParams {
    pub weights: Vec<i64>,
    pub biases: Vec<i64>,
}

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;


fn main() {
    println!("Gemini Benchmarking");
    println!("----------------------------------");
    println!("deep_small_benches");
    println!("----------------------------------");
    bench_nn("FC_5_layers", "plonky2_deep_small_bench");
    bench_nn("FC_3_layers", "plonky2_deep_small_bench");
    bench_nn("FC_10_layers", "plonky2_deep_small_bench");
    bench_nn("FC_15_layers", "plonky2_deep_small_bench");
    bench_nn("FC_20_layers", "plonky2_deep_small_bench");
    bench_nn("FC_30_layers", "plonky2_deep_small_bench");
    bench_nn("FC_40_layers", "plonky2_deep_small_bench");
    println!("deep_benches");
    println!("----------------------------------");
    bench_nn("FC_50_layers", "plonky2_deep_bench");
    bench_nn("FC_75_layers", "plonky2_deep_bench");
    bench_nn("FC_100_layers", "plonky2_deep_bench");
    // // bench_nn("FC_150_layers", "plonky2_deep_bench");
    // // bench_nn("FC_200_layers", "plonky2_deep_bench");
    println!("num_params_benches");
    println!("----------------------------------");
    bench_nn("FC_11216646_params", "plonky2_num_params_bench");
    bench_nn("FC_1195804_params", "plonky2_num_params_bench");
    // // bench_nn("FC_14773663_params", "plonky2_num_params_bench");
    bench_nn("FC_2236477_params", "plonky2_num_params_bench");
    bench_nn("FC_4107399_params", "plonky2_num_params_bench");
    bench_nn("FC_517529_params", "plonky2_num_params_bench");
    bench_nn("FC_676836_params", "plonky2_num_params_bench");
    bench_nn("FC_7770136_params", "plonky2_num_params_bench");
    println!("num_params_small_benches");
    println!("----------------------------------");
    bench_nn("FC_177522_params", "plonky2_num_params_small_bench");
    bench_nn("FC_48564_params", "plonky2_num_params_small_bench");
    bench_nn("FC_87771_params", "plonky2_num_params_small_bench")
}

fn bench_nn (network_name: &str, folder_parent: &str) {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::builder().testing().build();

    println!("Benches for {}", network_name);
    let rng = &mut test_rng();

    let (inputs, layers, outputs) = get_inputs(&format!("{}/{}.json", folder_parent, network_name));

    let cfg = CircomConfig::<Bls12_381>::new(format!("/home/ubuntu/circom/simple_circuit/bin/{}_js/{}.wasm", network_name, network_name), format!("/home/ubuntu/circom/simple_circuit/bin/{}.r1cs", network_name)).unwrap();
    let mut builder = CircomBuilder::new(cfg);
    
    for input in inputs {
        builder.push_input("in", input);
    }
    for (index, layer) in layers.iter().enumerate() {
        for weight in &layer.weights {
            builder.push_input(format!("w{}", index), *weight);
        }
        for bias in &layer.biases {
            builder.push_input(format!("b{}", index), *bias);
        }
    }
    // builder.push_input("a", 5);
    // builder.push_input("b", 10);
    let circom = builder.build().unwrap();
    let inputs_gen = circom.get_public_inputs().unwrap();
    //println!("public inputs are: {:?}", inputs_gen.iter().map(|x| x.to_string()).collect::<Vec<_>>());
    println!("gate count is {:?}", circom.r1cs.constraints.len());
    let r1cs = generate_relation(circom);
    let max_degree = r1cs.w.len();

    let now = Instant::now();

    let ck = ark_gemini::kzg::CommitterKey::new(max_degree, 5, rng);
    println!("ck built in {:?}", now.elapsed().as_secs());

    let now = Instant::now();

    let proof = Proof::new_time(&r1cs, &ck);
    println!("Proof is done in {:?}", now.elapsed().as_secs());

    #[cfg(feature = "dhat-heap")]
    {
        let stats = dhat::HeapStats::get();
        println!("bytes used: {:?}", stats.max_bytes);
    }
    //let vk = ark_gemini::kzg::VerifierKey::from(&ck);
    //proof.verify(&r1cs, &vk);
    //println!("verification is done!");
}

fn get_inputs(file_path: &str) -> (Vec<i64>, Vec<LayerParams>, Vec<i64>) {
    //const PREFIX: &str = "/home/aweso/arkworks/gemini_test/";
    const PREFIX: &str = "/home/ubuntu/plonky2_bench/bench_objects/";
    let inputs_raw = std::fs::read_to_string(PREFIX.to_owned() + file_path)
    .unwrap();
    let inputs = json::parse(&inputs_raw).unwrap();
    let input: Vec<_> = inputs["input"]
        .members()
        .map(|x| x.as_i64().unwrap())
        .collect();
    let layers: Vec<LayerParams> = inputs["layers"]
        .members()
        .map(|layer| LayerParams {
            weights: layer["weight"]
                .members()
                .map(|x| x.as_i64().unwrap())
                .collect(),
            biases: layer["bias"]
                .members()
                .map(|x| x.as_i64().unwrap())
                .collect(),
        })
        .collect();

    let output: Vec<_> = inputs["output"]
        .members()
        .map(|x| x.as_i64().unwrap())
        .collect();

    (input, layers, output)
}
