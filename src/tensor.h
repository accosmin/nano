#pragma once

#include "scalar.h"
#include "tensor/tensor.hpp"

namespace cortex
{
        using vector_t = tensor::vector_t<scalar_t>;
        using vectors_t = std::vector<vector_t>;

        using matrix_t = tensor::matrix_t<scalar_t>;
        using matrices_t = std::vector<matrix_t>;

        using tensor_t = tensor::tensor_t<scalar_t>;
        using tensors_t = std::vector<tensor_t>;

        using tensor_size_t = tensor_t::Index;
        using tensor_index_t = tensor_t::Index;
}


