#pragma once

#include "scalar.h"
#include "nanocv/tensor/tensor.hpp"

namespace ncv
{
        // numerical types (default)
        typedef tensor::vector_types_t<scalar_t>::tvector       vector_t;
        typedef tensor::vector_types_t<scalar_t>::tvectors      vectors_t;

        typedef tensor::matrix_types_t<scalar_t>::tmatrix       matrix_t;
        typedef tensor::matrix_types_t<scalar_t>::tmatrices     matrices_t;

        typedef tensor::tensor_t<scalar_t>                      tensor_t;
        typedef std::vector<tensor_t>                           tensors_t;

        // numerical types (float - for speed/vectorization)
        typedef tensor::vector_types_t<float>::tvector          fvector_t;
        typedef tensor::vector_types_t<float>::tvectors         fvectors_t;

        typedef tensor::matrix_types_t<float>::tmatrix          fmatrix_t;
        typedef tensor::matrix_types_t<float>::tmatrices        fmatrices_t;

        typedef tensor::tensor_t<float>                         ftensor_t;
        typedef std::vector<ftensor_t>                          ftensors_t;
}


