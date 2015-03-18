#pragma once

#include "tensor/tensor.hpp"
#include "util/text.h"
#include <functional>

namespace ncv
{
        // numerical types
        typedef std::size_t                                     size_t;
        typedef std::vector<size_t>                             indices_t;

#ifdef NANOCV_WITH_FLOAT
        typedef float                                           scalar_t;
#elseif MANOCV_WITH_DOUBLE
        typedef double                                          scalar_t;
#elseif NANOCV_WIDTH_LONG_DOUBLE
        typedef long double                                     scalar_t;
#else
        typedef double                                          scalar_t;
#endif
        typedef std::vector<scalar_t>                           scalars_t;

        typedef tensor::vector_types_t<scalar_t>::tvector       vector_t;
        typedef tensor::vector_types_t<scalar_t>::tvectors      vectors_t;

        typedef tensor::matrix_types_t<scalar_t>::tmatrix       matrix_t;
        typedef tensor::matrix_types_t<scalar_t>::tmatrices     matrices_t;

        typedef tensor::tensor_t<scalar_t, size_t>              tensor_t;
        typedef std::vector<tensor_t>                           tensors_t;

        // strings
        typedef std::string                                     string_t;
        typedef std::vector<string_t>                           strings_t;

        // lambda
        using std::placeholders::_1;
        using std::placeholders::_2;
        using std::placeholders::_3;
        using std::placeholders::_4;

        ///
        /// \brief color channels
        ///
        enum class color_channel
        {
                red = 0,                // R
                green,                  // G
                blue,                   // B
                luma,                   // Y/L
                rgba,                   // RGBA
                cielab_l,               // CIELab L
                cielab_a,               // CIELab a
                cielab_b                // CIELab b
        };

        ///
        /// \brief color processing mode methods
        ///
        enum class color_mode
        {
                luma,                   ///< process only grayscale color channel
                rgba                    ///< process red, green & blue color channels
        };

        // string cast for enumerations
        namespace text
        {
                template <>
                inline string_t to_string(color_mode mode)
                {
                        switch (mode)
                        {
                        case color_mode::luma:          return "luma";
                        case color_mode::rgba:          return "rgba";
                        default:                        return "luma";
                        }
                }

                template <>
                inline color_mode from_string<color_mode>(const string_t& string)
                {
                        if (string == "luma")           return color_mode::luma;
                        if (string == "rgba")           return color_mode::rgba;
                        throw std::invalid_argument("invalid color mode <" + string + ">!");
                        return color_mode::luma;
                }

                template <>
                inline string_t to_string(color_channel type)
                {
                        switch (type)
                        {
                        case color_channel::red:        return "red";
                        case color_channel::green:      return "green";
                        case color_channel::blue:       return "blue";
                        case color_channel::luma:       return "luma";
                        case color_channel::rgba:       return "rgba";
                        case color_channel::cielab_l:   return "cielab_l";
                        case color_channel::cielab_a:   return "cielab_a";
                        case color_channel::cielab_b:   return "cielab_b";
                        default:                        return "luma";
                        }
                }

                template <>
                inline color_channel from_string<color_channel>(const string_t& string)
                {
                        if (string == "red")            return color_channel::red;
                        if (string == "green")          return color_channel::green;
                        if (string == "blue")           return color_channel::blue;
                        if (string == "luma")           return color_channel::luma;
                        if (string == "rgba")           return color_channel::rgba;
                        if (string == "cielab_l")       return color_channel::cielab_l;
                        if (string == "cielab_a")       return color_channel::cielab_a;
                        if (string == "cielab_b")       return color_channel::cielab_b;
                        throw std::invalid_argument("Invalid color channel <" + string + ">!");
                        return color_channel::luma;
                }
        }
}


