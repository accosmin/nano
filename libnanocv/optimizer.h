#pragma once

#include "optimize/problem.hpp"
#include "optimize/linesearch.h"
#include "text.h"
#include "tensor.h"

namespace ncv
{
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
        >                                                       opt_problem_t;

        typedef opt_problem_t::tstate                           opt_state_t;

        typedef opt_problem_t::twlog                            opt_opwlog_t;
        typedef opt_problem_t::telog                            opt_opelog_t;
        typedef opt_problem_t::tulog                            opt_opulog_t;

        ///
        /// \brief stochastic optimization methods
        ///
        enum class stochastic_optimizer
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
        enum class batch_optimizer
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
                CGD_N,
                CGD_DYCD,
                CGD_DYHS
        };

        // string cast for enumerations
        namespace text
        {
                template <>
                inline std::string to_string(stochastic_optimizer type)
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
                inline stochastic_optimizer from_string<stochastic_optimizer>(const std::string& string)
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
                inline std::string to_string(batch_optimizer type)
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
                        case batch_optimizer::CGD_DYCD: return "cgd-dycd";
                        case batch_optimizer::CGD_DYHS: return "cgd-dyhs";
                        default:                        return "lbfgs";
                        }
                }

                template <>
                inline batch_optimizer from_string<batch_optimizer>(const std::string& string)
                {
                        if (string == "gd")             return batch_optimizer::GD;
                        if (string == "cgd")            return batch_optimizer::CGD;
                        if (string == "lbfgs")          return batch_optimizer::LBFGS;
                        if (string == "cgd-hs")         return batch_optimizer::CGD_HS;
                        if (string == "cgd-fr")         return batch_optimizer::CGD_FR;
                        if (string == "cgd-pr")         return batch_optimizer::CGD_PR;
                        if (string == "cgd-cd")         return batch_optimizer::CGD_CD;
                        if (string == "cgd-ls")         return batch_optimizer::CGD_LS;
                        if (string == "cgd-dy")         return batch_optimizer::CGD_DY;
                        if (string == "cgd-n")          return batch_optimizer::CGD_N;
                        if (string == "cgd-dycd")       return batch_optimizer::CGD_DYCD;
                        if (string == "cgd-dyhs")       return batch_optimizer::CGD_DYHS;
                        throw std::invalid_argument("invalid batch optimizer <" + string + ">!");
                        return batch_optimizer::GD;
                }

                template <>
                inline std::string to_string(optimize::ls_initializer type)
                {
                        switch (type)
                        {
                        case optimize::ls_initializer::unit:            return "init-unit";
                        case optimize::ls_initializer::quadratic:       return "init-quadratic";
                        case optimize::ls_initializer::consistent:      return "init-consistent";
                        default:                                        return "none";
                        }
                }

                template <>
                inline std::string to_string(optimize::ls_strategy type)
                {
                        switch (type)
                        {
                        case optimize::ls_strategy::backtrack_armijo:   return "backtrack-Armijo";
                        case optimize::ls_strategy::backtrack_wolfe:    return "backtrack-Wolfe";
                        case optimize::ls_strategy::backtrack_strong_wolfe:     return "backtrack-strong-Wolfe";
                        case optimize::ls_strategy::interpolation_bisection:    return "interp-bisection";
                        case optimize::ls_strategy::interpolation_cubic:        return "interp-cubic";
                        default:                                        return "none";
                        }
                }
        }
}


