#![feature(portable_simd, associated_type_defaults, generic_const_exprs, inherent_associated_types, auto_traits, negative_impls)]


extern crate core;

mod matrices;
mod meta;


#[derive(Debug)]
pub enum MathError {
    NonSquareMatrix,
    NonInvertibleMatrix,
}

pub type MathResult<T> = Result<T, MathError>;