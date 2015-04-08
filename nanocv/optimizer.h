#pragma once

#include "text.h"
#include "tensor.h"
#include "optim/problem.hpp"
#include "optim/types.h"

namespace ncv
{
        // optimization data types
        typedef std::function<size_t(void)>                             opt_opsize_t;
        typedef std::function<scalar_t(const vector_t&)>                opt_opfval_t;
        typedef std::function<scalar_t(const vector_t&, vector_t&)>     opt_opgrad_t;

        typedef optim::problem_t
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

        // string cast for enumerations
        namespace text
        {
                template <>
                inline std::string to_string(optim::stoch_optimizer type)
                {
                        switch (type)
                        {
                        case optim::stoch_optimizer::SG:           return "sg";
                        case optim::stoch_optimizer::SGA:          return "sga";
                        case optim::stoch_optimizer::SIA:          return "sia";
                        case optim::stoch_optimizer::AG:           return "ag";
                        case optim::stoch_optimizer::ADAGRAD:      return "adagrad";
                        case optim::stoch_optimizer::ADADELTA:     return "adadelta";
                        default:                                        return "????";
                        }
                }

                template <>
                inline optim::stoch_optimizer from_string<optim::stoch_optimizer>(const std::string& string)
                {
                        if (string == "sg")             return optim::stoch_optimizer::SG;
                        if (string == "sga")            return optim::stoch_optimizer::SGA;
                        if (string == "sia")            return optim::stoch_optimizer::SIA;
                        if (string == "ag")             return optim::stoch_optimizer::AG;
                        if (string == "adagrad")        return optim::stoch_optimizer::ADAGRAD;
                        if (string == "adadelta")       return optim::stoch_optimizer::ADADELTA;
                        throw std::invalid_argument("invalid stochastic optimizer <" + string + ">!");
                        return optim::stoch_optimizer::SG;
                }

                template <>
                inline std::string to_string(optim::batch_optimizer type)
                {
                        switch (type)
                        {
                        case optim::batch_optimizer::GD:                return "gd";
                        case optim::batch_optimizer::CGD:               return "cgd";
                        case optim::batch_optimizer::LBFGS:             return "lbfgs";
                        case optim::batch_optimizer::CGD_HS:            return "cgd-hs";
                        case optim::batch_optimizer::CGD_FR:            return "cgd-fr";
                        case optim::batch_optimizer::CGD_PRP:           return "cgd-prp";
                        case optim::batch_optimizer::CGD_CD:            return "cgd-cd";
                        case optim::batch_optimizer::CGD_LS:            return "cgd-ls";
                        case optim::batch_optimizer::CGD_DY:            return "cgd-dy";
                        case optim::batch_optimizer::CGD_N:             return "cgd-n";
                        case optim::batch_optimizer::CGD_DYCD:          return "cgd-dycd";
                        case optim::batch_optimizer::CGD_DYHS:          return "cgd-dyhs";
                        default:                                        return "????";
                        }
                }

                template <>
                inline optim::batch_optimizer from_string<optim::batch_optimizer>(const std::string& string)
                {
                        if (string == "gd")             return optim::batch_optimizer::GD;
                        if (string == "cgd")            return optim::batch_optimizer::CGD_N;
                        if (string == "lbfgs")          return optim::batch_optimizer::LBFGS;
                        if (string == "cgd-hs")         return optim::batch_optimizer::CGD_HS;
                        if (string == "cgd-fr")         return optim::batch_optimizer::CGD_FR;
                        if (string == "cgd-prp")        return optim::batch_optimizer::CGD_PRP;
                        if (string == "cgd-cd")         return optim::batch_optimizer::CGD_CD;
                        if (string == "cgd-ls")         return optim::batch_optimizer::CGD_LS;
                        if (string == "cgd-dy")         return optim::batch_optimizer::CGD_DY;
                        if (string == "cgd-n")          return optim::batch_optimizer::CGD_N;
                        if (string == "cgd-dycd")       return optim::batch_optimizer::CGD_DYCD;
                        if (string == "cgd-dyhs")       return optim::batch_optimizer::CGD_DYHS;
                        throw std::invalid_argument("invalid batch optimizer <" + string + ">!");
                        return optim::batch_optimizer::GD;
                }

                template <>
                inline std::string to_string(optim::ls_initializer type)
                {
                        switch (type)
                        {
                        case optim::ls_initializer::unit:               return "init-unit";
                        case optim::ls_initializer::quadratic:          return "init-quadratic";
                        case optim::ls_initializer::consistent:         return "init-consistent";
                        default:                                        return "????";
                        }
                }

                template <>
                inline std::string to_string(optim::ls_strategy type)
                {
                        switch (type)
                        {
                        case optim::ls_strategy::backtrack_armijo:              return "backtrack-Armijo";
                        case optim::ls_strategy::backtrack_wolfe:               return "backtrack-Wolfe";
                        case optim::ls_strategy::backtrack_strong_wolfe:        return "backtrack-strong-Wolfe";
                        case optim::ls_strategy::interpolation_bisection:       return "interp-bisection";
                        case optim::ls_strategy::interpolation_cubic:           return "interp-cubic";
                        case optim::ls_strategy::cg_descent:                    return "cgdescent";
                        default:                                                return "????";
                        }
                }
        }
}


