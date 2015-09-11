#pragma once

#include "scalar.h"
#include "libtensor/tensor.hpp"

namespace ncv
{
        // low-precision tensors
        typedef tensor::vector_t<lscalar_t>     lvector_t;
        typedef std::vector<lvector_t>          lvectors_t;

        typedef tensor::matrix_t<lscalar_t>     lmatrix_t;
        typedef std::vector<lmatrix_t>          lmatrices_t;

        typedef tensor::tensor_t<lscalar_t>     ltensor_t;
        typedef std::vector<ltensor_t>          ltensors_t;

        // high-precision tensors
        typedef tensor::vector_t<hscalar_t>     hvector_t;
        typedef std::vector<hvector_t>          hvectors_t;

        typedef tensor::matrix_t<hscalar_t>     hmatrix_t;
        typedef std::vector<hmatrix_t>          hmatrices_t;

        typedef tensor::tensor_t<hscalar_t>     htensor_t;
        typedef std::vector<htensor_t>          htensors_t;

        // default tensors
        typedef tensor::vector_t<scalar_t>      vector_t;
        typedef std::vector<vector_t>           vectors_t;

        typedef tensor::matrix_t<scalar_t>      matrix_t;
        typedef std::vector<matrix_t>           matrices_t;

        typedef tensor::tensor_t<scalar_t>      tensor_t;
        typedef std::vector<tensor_t>           tensors_t;
}


