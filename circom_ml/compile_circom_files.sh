#!/bin/bash

for i in 3 5 10 15 20 30 40
do
    echo FC_${i}_layers.circom
    circom ./networks/FC_${i}_layers.circom --r1cs --wasm -o "./bin/"
done