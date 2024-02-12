use crate::matrices::stack_matrix::StackMatrix;

// Common 3D graphics types
pub type Vector2<T> = StackMatrix<T, 1, 2>;
pub type Vector2Transposed<T> = StackMatrix<T, 2, 1>;
pub type Vector3<T> = StackMatrix<T, 1, 3>;
pub type Vector3Transposed<T> = StackMatrix<T, 3, 1>;
pub type Vector4<T> = StackMatrix<T, 1, 4>;
pub type Vector4Transposed<T> = StackMatrix<T, 4, 1>;
pub type Matrix2x2<T> = StackMatrix<T, 2, 2>;
pub type Matrix2x3<T> = StackMatrix<T, 2, 3>;
pub type Matrix3x2<T> = StackMatrix<T, 3, 2>;
pub type Matrix3x3<T> = StackMatrix<T, 3, 3>;
pub type Matrix4x4<T> = StackMatrix<T, 4, 4>;
pub type Matrix3x4<T> = StackMatrix<T, 3, 4>;
pub type Matrix4x3<T> = StackMatrix<T, 4, 3>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices::matrix::Matrix;

    #[test]
    fn test_vector2() {
        let vec2: Vector2<u8> = [1, 2].into();
        assert_eq!(vec2.row_count(), 2);
        assert_eq!(vec2.column_count(), 1);
        assert_eq!(vec2[(0, 0)], 1);
        assert_eq!(vec2[(1, 0)], 2);
        assert_eq!(vec2[0], 1);
        assert_eq!(vec2[1], 2);
    }

    #[test]
    fn test_vector2_transposed() {
        let vec2: Vector2Transposed<u8> = [[1], [2]].into();
        assert_eq!(vec2.row_count(), 1);
        assert_eq!(vec2.column_count(), 2);
        assert_eq!(vec2[(0, 0)], 1);
        assert_eq!(vec2[(0, 1)], 2);
        assert_eq!(vec2[0], 1);
        assert_eq!(vec2[1], 2);

        let vec2_transposed: Vector2<u8> = vec2.transposed();
        assert_eq!(vec2_transposed, [1, 2].into());
    }

    #[test]
    fn test_vector3() {
        let vec3: Vector3<u8> = [1, 2, 3].into();
        assert_eq!(vec3.row_count(), 3);
        assert_eq!(vec3.column_count(), 1);
        assert_eq!(vec3[(0, 0)], 1);
        assert_eq!(vec3[(1, 0)], 2);
        assert_eq!(vec3[(2, 0)], 3);
        assert_eq!(vec3[0], 1);
        assert_eq!(vec3[1], 2);
        assert_eq!(vec3[2], 3);
    }

    #[test]
    fn test_vector3_transposed() {
        let vec3: Vector3Transposed<u8> = [[1], [2], [3]].into();
        assert_eq!(vec3.row_count(), 1);
        assert_eq!(vec3.column_count(), 3);
        assert_eq!(vec3[(0, 0)], 1);
        assert_eq!(vec3[(0, 1)], 2);
        assert_eq!(vec3[(0, 2)], 3);
        assert_eq!(vec3[0], 1);
        assert_eq!(vec3[1], 2);
        assert_eq!(vec3[2], 3);

        let vec3_transposed: Vector3<u8> = vec3.transposed();
        assert_eq!(vec3_transposed, [1, 2, 3].into());
    }

    #[test]
    fn test_vector4() {
        let vec4: Vector4<u8> = [1, 2, 3, 4].into();
        assert_eq!(vec4.row_count(), 4);
        assert_eq!(vec4.column_count(), 1);
        assert_eq!(vec4[(0, 0)], 1);
        assert_eq!(vec4[(1, 0)], 2);
        assert_eq!(vec4[(2, 0)], 3);
        assert_eq!(vec4[(3, 0)], 4);
        assert_eq!(vec4[0], 1);
        assert_eq!(vec4[1], 2);
        assert_eq!(vec4[2], 3);
        assert_eq!(vec4[3], 4);
    }

    #[test]
    fn test_vector4_transposed() {
        let vec4: Vector4Transposed<u8> = [[1], [2], [3], [4]].into();
        assert_eq!(vec4.row_count(), 1);
        assert_eq!(vec4.column_count(), 4);
        assert_eq!(vec4[(0, 0)], 1);
        assert_eq!(vec4[(0, 1)], 2);
        assert_eq!(vec4[(0, 2)], 3);
        assert_eq!(vec4[(0, 3)], 4);
        assert_eq!(vec4[0], 1);
        assert_eq!(vec4[1], 2);
        assert_eq!(vec4[2], 3);
        assert_eq!(vec4[3], 4);

        let vec4_transposed: Vector4<u8> = vec4.transposed();
        assert_eq!(vec4_transposed, [1, 2, 3, 4].into());
    }

    #[test]
    fn test_matrix_product() {
        let vec3: Vector3<u8> = [1, 2, 3].into();
        let vec3_transposed: Vector3Transposed<u8> = [[2], [3], [4]].into();
        let product = vec3.product::<3, 3>(&vec3_transposed);
    }
}
