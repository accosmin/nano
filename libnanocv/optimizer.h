#pragma once

#include "libnanocv/tensor.h"
#include "libmin/types.h"
#include "libmin/problem.hpp"
#include "libnanocv/text/enum_string.hpp"

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
                inline std::map<optim::stoch_optimizer, std::string> enum_string<optim::stoch_optimizer>()
                {
                        return
                        {
                                { optim::stoch_optimizer::SG,           "sg" },
                                { optim::stoch_optimizer::SGA,          "sga" },
                                { optim::stoch_optimizer::SIA,          "sia" },
                                { optim::stoch_optimizer::AG,           "ag" },
                                { optim::stoch_optimizer::AGGR,         "aggr" },
                                { optim::stoch_optimizer::ADAGRAD,      "adagrad" },
                                { optim::stoch_optimizer::ADADELTA,     "adadelta" }
                        };
                }

                template <>
                inline std::map<optim::batch_optimizer, std::string> enum_string<optim::batch_optimizer>()
                {
                        return
                        {
                                { optim::batch_optimizer::GD,           "gd" },
                                { optim::batch_optimizer::CGD,          "cgd" },
                                { optim::batch_optimizer::LBFGS,        "lbfgs" },
                                { optim::batch_optimizer::CGD_HS,       "cgd-hs" },
                                { optim::batch_optimizer::CGD_FR,       "cgd-fr" },
                                { optim::batch_optimizer::CGD_PRP,      "cgd-prp" },
                                { optim::batch_optimizer::CGD_CD,       "cgd-cd" },
                                { optim::batch_optimizer::CGD_LS,       "cgd-ls" },
                                { optim::batch_optimizer::CGD_DY,       "cgd-dy" },
                                { optim::batch_optimizer::CGD_N,        "cgd-n" },
                                { optim::batch_optimizer::CGD_DYCD,     "cgd-dycd" },
                                { optim::batch_optimizer::CGD_DYHS,     "cgd-dyhs" }
                        };
                }

                template <>
                inline std::map<optim::ls_initializer, std::string> enum_string<optim::ls_initializer>()
                {
                        return
                        {
                                { optim::ls_initializer::unit,          "init-unit" },
                                { optim::ls_initializer::quadratic,     "init-quadratic" },
                                { optim::ls_initializer::consistent,    "init-consistent" }
                        };
                }

                template <>
                inline std::map<optim::ls_strategy, std::string> enum_string<optim::ls_strategy>()
                {
                        return
                        {
                                { optim::ls_strategy::backtrack_armijo,         "backtrack-Armijo" },
                                { optim::ls_strategy::backtrack_wolfe,          "backtrack-Wolfe" },
                                { optim::ls_strategy::backtrack_strong_wolfe,   "backtrack-strong-Wolfe" },
                                { optim::ls_strategy::interpolation,            "interp" },
                                { optim::ls_strategy::cg_descent,               "cgdescent" }
                        };
                }
        }
}


