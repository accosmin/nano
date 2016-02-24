#pragma once

#include "scalar.h"
#include "math/problem.hpp"
#include "math/batch_types.h"
#include "math/stoch_types.h"
#include "math/lsearch_types.h"
#include "text/enum_string.hpp"

namespace cortex
{
        using opt_problem_t = math::problem_t<scalar_t>;

        using opt_size_t = opt_problem_t::tsize;
        using opt_state_t = opt_problem_t::tstate;

        using opt_opsize_t = opt_problem_t::topsize;
        using opt_opfval_t = opt_problem_t::topfval;
        using opt_opgrad_t = opt_problem_t::topgrad;
}

namespace text
{
        template <>
        inline std::map<math::stoch_optimizer, std::string> enum_string<math::stoch_optimizer>()
        {
                return
                {
                        { math::stoch_optimizer::SG,           "sg" },
                        { math::stoch_optimizer::SGM,          "sgm" },
                        { math::stoch_optimizer::AG,           "ag" },
                        { math::stoch_optimizer::AGFR,         "agfr" },
                        { math::stoch_optimizer::AGGR,         "aggr" },
                        { math::stoch_optimizer::ADAGRAD,      "adagrad" },
                        { math::stoch_optimizer::ADADELTA,     "adadelta" },
                        { math::stoch_optimizer::ADAM,         "adam" }
                };
        }

        template <>
        inline std::map<math::batch_optimizer, std::string> enum_string<math::batch_optimizer>()
        {
                return
                {
                        { math::batch_optimizer::GD,           "gd" },
                        { math::batch_optimizer::CGD,          "cgd" },
                        { math::batch_optimizer::LBFGS,        "lbfgs" },
                        { math::batch_optimizer::CGD_HS,       "cgd-hs" },
                        { math::batch_optimizer::CGD_FR,       "cgd-fr" },
                        { math::batch_optimizer::CGD_PRP,      "cgd-prp" },
                        { math::batch_optimizer::CGD_CD,       "cgd-cd" },
                        { math::batch_optimizer::CGD_LS,       "cgd-ls" },
                        { math::batch_optimizer::CGD_DY,       "cgd-dy" },
                        { math::batch_optimizer::CGD_N,        "cgd-n" },
                        { math::batch_optimizer::CGD_DYCD,     "cgd-dycd" },
                        { math::batch_optimizer::CGD_DYHS,     "cgd-dyhs" }
                };
        }

        template <>
        inline std::map<math::ls_initializer, std::string> enum_string<math::ls_initializer>()
        {
                return
                {
                        { math::ls_initializer::unit,          "init-unit" },
                        { math::ls_initializer::quadratic,     "init-quadratic" },
                        { math::ls_initializer::consistent,    "init-consistent" }
                };
        }

        template <>
        inline std::map<math::ls_strategy, std::string> enum_string<math::ls_strategy>()
        {
                return
                {
                        { math::ls_strategy::backtrack_armijo,         "backtrack-Armijo" },
                        { math::ls_strategy::backtrack_wolfe,          "backtrack-Wolfe" },
                        { math::ls_strategy::backtrack_strong_wolfe,   "backtrack-strong-Wolfe" },
                        { math::ls_strategy::interpolation,            "interp" },
                        { math::ls_strategy::cg_descent,               "cgdescent" }
                };
        }
}


