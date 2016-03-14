#pragma once

#include "scalar.h"
#include "tensor/tensor.hpp"

namespace nano
{
        using vector_t = tensor::vector_t<scalar_t>;
        using vectors_t = std::vector<vector_t>;

        using matrix_t = tensor::matrix_t<scalar_t>;
        using matrices_t = std::vector<matrix_t>;

        using tensor3d_t = tensor::tensor_t<scalar_t, 3>;
        using tensor3ds_t = std::vector<tensor3d_t>;

        using tensor4d_t = tensor::tensor_t<scalar_t, 4>;
        using tensor4ds_t = std::vector<tensor4d_t>;

        using tensor_size_t = tensor3d_t::Index;
        using tensor_index_t = tensor3d_t::Index;
}


