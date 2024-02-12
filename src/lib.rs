#![feature(repr128)]
#![feature(associated_type_defaults)]
#![feature(generic_const_exprs)]

extern crate core;

pub mod matrices;
pub mod meta;
pub mod tensors;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
