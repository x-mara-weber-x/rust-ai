trait Tensor<T> {}

// #[derive(Debug, Clone)]
// pub struct HeapTensor<T> {
//     cells: Vec<T>,
//     dimensionality: Vec<usize>,
// }
//
// impl<T> HeapTensor<T>
// where
//     T: Copy + Default,
// {
//     fn new(dimensionality: Vec<usize>) -> Self {
//         Self {
//             cells: vec![Default::default(); row_count * column_count],
//             row_count,
//             column_count,
//         }
//     }
// }
//
// pub struct FlowTensor<T> {}
//
// impl Tensor<f32> for FlowTensor<f32> {}
