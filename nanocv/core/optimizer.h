#pragma once

#include "tensor.h"
#include "min/types.h"
#include "min/problem.hpp"
#include "text/enum_string.hpp"

namespace ncv
{
        // optimization data types
        typedef std::function<size_t(void)>                             opt_opsize_t;
        typedef std::function<scalar_t(const vector_t&)>                opt_opfval_t;
        typedef std::function<scalar_t(const vector_t&, vector_t&)>     opt_opgrad_t;

        typedef min::problem_t
        <
                scalar_t,
                size_t,
                opt_opsize_t,
                opt_opfval_t,
                opt_opgrad_t
        >                                                       opt_problem_t;

        typedef opt_problem_t::tstate                           opt_state_t;
        typedef opt_problem_t::tulog                            opt_opulog_t;

        // string cast for enumerations
        namespace text
        {
                template <>
                inline std::map<min::stoch_optimizer, std::string> enum_string<min::stoch_optimizer>()
                {
                        return
                        {
                                { min::stoch_optimizer::SG,           "sg" },
                                { min::stoch_optimizer::SGA,          "sga" },
                                { min::stoch_optimizer::SIA,          "sia" },
                                { min::stoch_optimizer::AG,           "ag" },
                                { min::stoch_optimizer::AGGR,         "aggr" },
                                { min::stoch_optimizer::ADAGRAD,      "adagrad" },
                                { min::stoch_optimizer::ADADELTA,     "adadelta" }
                        };
                }

                template <>
                inline std::map<min::batch_optimizer, std::string> enum_string<min::batch_optimizer>()
                {
                        return
                        {
                                { min::batch_optimizer::GD,           "gd" },
                                { min::batch_optimizer::CGD,          "cgd" },
                                { min::batch_optimizer::LBFGS,        "lbfgs" },
                                { min::batch_optimizer::CGD_HS,       "cgd-hs" },
                                { min::batch_optimizer::CGD_FR,       "cgd-fr" },
                                { min::batch_optimizer::CGD_PRP,      "cgd-prp" },
                                { min::batch_optimizer::CGD_CD,       "cgd-cd" },
                                { min::batch_optimizer::CGD_LS,       "cgd-ls" },
                                { min::batch_optimizer::CGD_DY,       "cgd-dy" },
                                { min::batch_optimizer::CGD_N,        "cgd-n" },
                                { min::batch_optimizer::CGD_DYCD,     "cgd-dycd" },
                                { min::batch_optimizer::CGD_DYHS,     "cgd-dyhs" }
                        };
                }

                template <>
                inline std::map<min::ls_initializer, std::string> enum_string<min::ls_initializer>()
                {
                        return
                        {
                                { min::ls_initializer::unit,          "init-unit" },
                                { min::ls_initializer::quadratic,     "init-quadratic" },
                                { min::ls_initializer::consistent,    "init-consistent" }
                        };
                }

                template <>
                inline std::map<min::ls_strategy, std::string> enum_string<min::ls_strategy>()
                {
                        return
                        {
                                { min::ls_strategy::backtrack_armijo,         "backtrack-Armijo" },
                                { min::ls_strategy::backtrack_wolfe,          "backtrack-Wolfe" },
                                { min::ls_strategy::backtrack_strong_wolfe,   "backtrack-strong-Wolfe" },
                                { min::ls_strategy::interpolation,            "interp" },
                                { min::ls_strategy::cg_descent,               "cgdescent" }
                        };
                }
        }
}


