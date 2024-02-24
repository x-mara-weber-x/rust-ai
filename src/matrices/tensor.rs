pub trait Tensor<T> {
    fn new(dimensions: &[u64]) -> Self;
}
