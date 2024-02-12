use crate::matrices::aliases::Vector2;
use crate::matrices::matrix::Matrix;

fn transpose_matrix<TMatrix: Matrix>(matrix: TMatrix) -> TMatrix::Transposed {
    matrix.transposed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices::aliases::Vector2Transposed;

    #[test]
    fn test_transpose_matrix() {
        let vec2: Vector2<u8> = [1, 2].into();
        let vec2_transposed: Vector2Transposed<u8> = transpose_matrix(vec2);
    }
}
