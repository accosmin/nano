#ifndef NANOCV_TYPES_H
#define NANOCV_TYPES_H

#include "tensor/tensor.hpp"
#include "tensor/vectorizer.hpp"
#include "optimize/problem.hpp"
#include "common/text.h"
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

        // string cast for enumerations
        namespace text
        {
                template <>
                inline std::string to_string(protocol type)
                {
                        switch (type)
                        {
                        case protocol::train:           return "train";
                        case protocol::test:            return "test";
                        default:                        return "train";
                        }
                }

                template <>
                inline protocol from_string<protocol>(const std::string& string)
                {
                        if (string == "train")          return protocol::train;
                        if (string == "test")           return protocol::test;
                        throw std::invalid_argument("invalid protocol type <" + string + ">!");
                        return protocol::train;
                }

                template <>
                inline std::string to_string(color_mode mode)
                {
                        switch (mode)
                        {
                        case color_mode::luma:          return "luma";
                        case color_mode::rgba:          return "rgba";
                        default:                        return "luma";
                        }
                }

                template <>
                inline color_mode from_string<color_mode>(const std::string& string)
                {
                        if (string == "luma")           return color_mode::luma;
                        if (string == "rgba")           return color_mode::rgba;
                        throw std::invalid_argument("invalid color mode <" + string + ">!");
                        return color_mode::luma;
                }

                template <>
                inline std::string to_string(channel dtype)
                {
                        switch (dtype)
                        {
                        case channel::red:              return "red";
                        case channel::green:            return "green";
                        case channel::blue:             return "blue";
                        case channel::luma:             return "luma";
                        case channel::cielab_l:         return "cielab_l";
                        case channel::cielab_a:         return "cielab_a";
                        case channel::cielab_b:         return "cielab_b";
                        default:                        return "luma";
                        }
                }

                template <>
                inline channel from_string<channel>(const std::string& string)
                {
                        if (string == "red")            return channel::red;
                        if (string == "green")          return channel::green;
                        if (string == "blue")           return channel::blue;
                        if (string == "luma")           return channel::luma;
                        if (string == "cielab_l")       return channel::cielab_l;
                        if (string == "cielab_a")       return channel::cielab_a;
                        if (string == "cielab_b")       return channel::cielab_b;
                        throw std::invalid_argument("Invalid channel type <" + string + ">!");
                        return channel::luma;
                }
        }

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

