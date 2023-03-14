use plonky2::{field::extension::Extendable, hash::hash_types::RichField};

pub mod nn_ops;

pub fn felt_from_i64<F: RichField + Extendable<D>, const D: usize>(x: i64) -> F {
    assert!(
        x.unsigned_abs() < (F::NEG_ONE / F::TWO).to_canonical_u64(),
        "integers are too large!"
    );
    if x.is_positive() {
        F::from_canonical_u64(x.unsigned_abs())
    } else {
        F::from_canonical_u64(x.unsigned_abs()).neg()
    }
}
pub fn i64_from_felt<F: RichField + Extendable<D>, const D: usize>(x: F) -> i64 {
    if x.to_canonical_u64() < (F::NEG_ONE / F::TWO).to_canonical_u64() {
        i64::try_from(x.to_canonical_u64()).unwrap()
    } else {
        -i64::try_from(x.neg().to_canonical_u64()).unwrap()
    }
}
