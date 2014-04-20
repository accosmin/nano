#ifndef NANOCV_TYPES_H
#define NANOCV_TYPES_H

#include "tensor/tensor.hpp"
#include "tensor/vectorizer.hpp"
#include "optimize/problem.hpp"
#include <string>
#include <cstdint>

namespace ncv
{
        // numerical types
        typedef std::size_t                                     size_t;
        typedef std::vector<size_t>                             indices_t;

        typedef double                                          scalar_t;
        typedef std::vector<scalar_t>                           scalars_t;

        typedef tensor::vector_types_t<scalar_t>::tvector       vector_t;
        typedef tensor::vector_types_t<scalar_t>::tvectors      vectors_t;

        typedef tensor::matrix_types_t<scalar_t>::tmatrix       matrix_t;
        typedef tensor::matrix_types_t<scalar_t>::tmatrices     matrices_t;

        typedef tensor::tensor_t<scalar_t, size_t>              tensor_t;
        typedef std::vector<tensor_t>                           tensors_t;

        typedef tensor::ivectorizer_t<scalar_t, size_t>         ivectorizer_t;
        typedef tensor::ovectorizer_t<scalar_t, size_t>         ovectorizer_t;

        // strings
        typedef std::string                                     string_t;
        typedef std::vector<string_t>                           strings_t;

        // lambda
        using std::placeholders::_1;
        using std::placeholders::_2;
        using std::placeholders::_3;
        using std::placeholders::_4;

        // alignment options
        enum class align : int
        {
                left,
                center,
                right
        };

        // color channels
        enum class channel : int
        {
                red = 0,                // R
                green,                  // G
                blue,                   // B
                luma,                   // Y/L
                cielab_l,               // CIELab L
                cielab_a,               // CIELab a
                cielab_b                // CIELab b
        };

        // machine learning protocol
        enum class protocol : int
        {
                train = 0,              // training
                test                    // testing
        };

        // color processing mode method
        enum class color_mode : int
        {
                luma,                   // process only grayscale color channel
                rgba                    // process red, green & blue color channels
        };

        // optimization data types
        typedef std::function<size_t(void)>                             opt_opsize_t;
        typedef std::function<scalar_t(const vector_t&)>                opt_opfval_t;
        typedef std::function<scalar_t(const vector_t&, vector_t&)>     opt_opgrad_t;

        typedef optimize::problem_t
        <
                scalar_t,
                size_t,
                opt_opsize_t,
                opt_opfval_t,
                opt_opgrad_t
        >                                                               opt_problem_t;

        typedef opt_problem_t::tstate                                   opt_state_t;

        typedef opt_problem_t::twlog                                    opt_opwlog_t;
        typedef opt_problem_t::telog                                    opt_opelog_t;
        typedef opt_problem_t::tulog                                    opt_opulog_t;
}

#endif // NANOCV_TYPES_H

