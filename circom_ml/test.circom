include "/home/aweso/circom/simple_circuit/fc_layer/fc_layer.circom";
template network(in_len, out_len) {
    signal input in[in_len];
    signal output out[out_len];

    component l0 = fc(32, 100);
    signal input w0[100][32];
    signal input b0[100];
    l0.weights <== w0;
    l0.biases <== b0;
    l0.in <== in;
    component l1 = fc(100, 200);
    signal input w1[200][100];
    signal input b1[200];
    l1.weights <== w1;
    l1.biases <== b1;
    l1.in <== l0.out;
    component l2 = fc_no_relu(200, 100);
    signal input w2[100][200];
    signal input b2[100];
    l2.weights <== w2;
    l2.biases <== b2;
    l2.in <== l1.out;
    out <== l2.out;
}

component main = network(32, 100);