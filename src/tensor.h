#pragma once

#include "scalar.h"
#include "tensor/tensor.h"

namespace nano
{
        using vector_t = tensor_vector_t<scalar_t>;
        using matrix_t = tensor_matrix_t<scalar_t>;
        using tensor3d_t = tensor_mem_t<vector_t, 3>;
        using tensor4d_t = tensor_mem_t<vector_t, 4>;

        using vector_map_t = Eigen::Map<vector_t>;
        using tensor3d_map_t = tensor_map_t<vector_t, 3>;
        using tensor4d_map_t = tensor_map_t<vector_t, 4>;

        using vectors_t = std::vector<vector_t>;
        using matrices_t = std::vector<matrix_t>;
        using tensor3ds_t = std::vector<tensor3d_t>;
        using tensor4ds_t = std::vector<tensor4d_t>;

        using dim3d_t = tensor3d_t::tdims;
        using dim4d_t = tensor4d_t::tdims;
}
