#pragma once

#include "libnanocv/tensor/tensor.hpp"
#include "optimize/problem.hpp"
#include "libnanocv/util/text.h"
#include <cstdint>

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
        enum class color_channel : int
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
        enum class color_mode : int
        {
                luma,                   ///< process only grayscale color channel
                rgba                    ///< process red, green & blue color channels
        };

        ///
        /// \brief machine learning protocols
        ///
        enum class protocol : int
        {
                train = 0,              ///< training
                test                    ///< testing
        };

        ///
        /// \brief stochastic optimization methods
        ///
        enum class stochastic_optimizer : int
        {
                SG,                     ///< stochastic gradient
                SGA,                    ///< stochastic gradient averaging
                SIA,                    ///< stochastic iterate averaging
                AG,                     ///< Nesterov's accelerated gradient descent
                ADAGRAD,                ///< AdaGrad
                ADADELTA                ///< AdaDelta
        };

        ///
        /// \brief batch optimization methods
        ///
        enum class batch_optimizer : int
        {
                GD,                     ///< gradient descent
                CGD,                    ///< conjugate gradient descent (default version)
                LBFGS,                  ///< limited-memory BFGS

                CGD_HS,                 ///< various conjugate gradient descent versions
                CGD_FR,
                CGD_PR,
                CGD_CD,
                CGD_LS,
                CGD_DY,
                CGD_N
        };

        ///
        /// \brief regularization methods
        ///
        enum class regularizer : int
        {
                none = 0,               ///< no regularization term
                l2norm,                 ///< L2-norm regularization
                variational             ///< penalize high loss variation
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

                template <>
                inline string_t to_string(protocol type)
                {
                        switch (type)
                        {
                        case protocol::train:           return "train";
                        case protocol::test:            return "test";
                        default:                        return "train";
                        }
                }

                template <>
                inline protocol from_string<protocol>(const string_t& string)
                {
                        if (string == "train")          return protocol::train;
                        if (string == "test")           return protocol::test;
                        throw std::invalid_argument("invalid protocol <" + string + ">!");
                        return protocol::train;
                }

                template <>
                inline string_t to_string(stochastic_optimizer type)
                {
                        switch (type)
                        {
                        case stochastic_optimizer::SG:  return "sg";
                        case stochastic_optimizer::SGA: return "sga";
                        case stochastic_optimizer::SIA: return "sia";
                        case stochastic_optimizer::AG:  return "ag";
                        case stochastic_optimizer::ADAGRAD: return "adagrad";
                        case stochastic_optimizer::ADADELTA: return "adadelta";
                        default:                        return "sg";
                        }
                }

                template <>
                inline stochastic_optimizer from_string<stochastic_optimizer>(const string_t& string)
                {
                        if (string == "sg")             return stochastic_optimizer::SG;
                        if (string == "sga")            return stochastic_optimizer::SGA;
                        if (string == "sia")            return stochastic_optimizer::SIA;
                        if (string == "ag")             return stochastic_optimizer::AG;
                        if (string == "adagrad")        return stochastic_optimizer::ADAGRAD;
                        if (string == "adadelta")       return stochastic_optimizer::ADADELTA;
                        throw std::invalid_argument("invalid stochastic optimizer <" + string + ">!");
                        return stochastic_optimizer::SG;
                }

                template <>
                inline string_t to_string(batch_optimizer type)
                {
                        switch (type)
                        {
                        case batch_optimizer::GD:       return "gd";
                        case batch_optimizer::CGD:      return "cgd";
                        case batch_optimizer::LBFGS:    return "lbfgs";
                        case batch_optimizer::CGD_HS:   return "cgd-hs";
                        case batch_optimizer::CGD_FR:   return "cgd-fr";
                        case batch_optimizer::CGD_PR:   return "cgd-pr";
                        case batch_optimizer::CGD_CD:   return "cgd-cd";
                        case batch_optimizer::CGD_LS:   return "cgd-ls";
                        case batch_optimizer::CGD_DY:   return "cgd-dy";
                        case batch_optimizer::CGD_N:    return "cgd-n";
                        default:                        return "lbfgs";
                        }
                }

                template <>
                inline batch_optimizer from_string<batch_optimizer>(const string_t& string)
                {
                        if (string == "gd")             return batch_optimizer::GD;
                        if (string == "cgd")            return batch_optimizer::CGD;
                        if (string == "lbfgs")          return batch_optimizer::LBFGS;
                        throw std::invalid_argument("invalid batch optimizer <" + string + ">!");
                        return batch_optimizer::GD;
                }

                template <>
                inline string_t to_string(regularizer type)
                {
                        switch (type)
                        {
                        case regularizer::none:         return "none";
                        case regularizer::l2norm:       return "l2";
                        case regularizer::variational:  return "var";
                        default:                        return "none";
                        }
                }

                template <>
                inline regularizer from_string<regularizer>(const string_t& string)
                {
                        if (string == "none")           return regularizer::none;
                        if (string == "l2")             return regularizer::l2norm;
                        if (string == "var")            return regularizer::variational;
                        throw std::invalid_argument("invalid regularizer <" + string + ">!");
                        return regularizer::none;
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


