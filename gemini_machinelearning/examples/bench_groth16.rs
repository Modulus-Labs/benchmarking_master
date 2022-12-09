use ark_std::{test_rng, start_timer, end_timer};

use r1cs_machinelearning::ark_circom_mini::{CircomBuilder, CircomConfig};
use ark_groth16::{create_random_proof as prove, generate_random_parameters};
use ark_bn254::Bn254;


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
    let mut rng = &mut test_rng();

    let (inputs, layers, outputs) = get_inputs(&format!("{}/{}.json", folder_parent, network_name));

    let witness_timer = start_timer!(|| "build witness");
    let cfg = CircomConfig::<Bn254>::new(format!("/home/ubuntu/circom/simple_circuit/bin/{}_js/{}.wasm", network_name, network_name), format!("/home/ubuntu/circom/simple_circuit/bin/{}.r1cs", network_name)).unwrap();
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
    let circom = builder.setup();


    let params_timer = start_timer!(|| "build parameters");

    let params = generate_random_parameters::<Bn254, _, _>(circom, &mut rng).unwrap();

    end_timer!(params_timer);

    let circom = builder.build().unwrap();

    end_timer!(witness_timer);

    let proof_timer = start_timer!(|| "prove circuit");

    let proof = prove(circom, &params, &mut rng);

    end_timer!(proof_timer);

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