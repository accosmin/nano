#pragma once

#include "scalar.h"
#include "math/problem.hpp"
#include "math/batch_types.h"
#include "math/stoch_types.h"
#include "math/lsearch_types.h"
#include "text/enum_string.hpp"

namespace zob
{
        using opt_problem_t = zob::problem_t<scalar_t>;

        using opt_size_t = opt_problem_t::tsize;
        using opt_state_t = opt_problem_t::tstate;

        using opt_opsize_t = opt_problem_t::topsize;
        using opt_opfval_t = opt_problem_t::topfval;
        using opt_opgrad_t = opt_problem_t::topgrad;
}

namespace zob
{
        template <>
        inline std::map<zob::stoch_optimizer, std::string> enum_string<zob::stoch_optimizer>()
        {
                return
                {
                        { zob::stoch_optimizer::SG,           "sg" },
                        { zob::stoch_optimizer::SGM,          "sgm" },
                        { zob::stoch_optimizer::AG,           "ag" },
                        { zob::stoch_optimizer::AGFR,         "agfr" },
                        { zob::stoch_optimizer::AGGR,         "aggr" },
                        { zob::stoch_optimizer::ADAGRAD,      "adagrad" },
                        { zob::stoch_optimizer::ADADELTA,     "adadelta" },
                        { zob::stoch_optimizer::ADAM,         "adam" }
                };
        }

        template <>
        inline std::map<zob::batch_optimizer, std::string> enum_string<zob::batch_optimizer>()
        {
                return
                {
                        { zob::batch_optimizer::GD,           "gd" },
                        { zob::batch_optimizer::CGD,          "cgd" },
                        { zob::batch_optimizer::LBFGS,        "lbfgs" },
                        { zob::batch_optimizer::CGD_HS,       "cgd-hs" },
                        { zob::batch_optimizer::CGD_FR,       "cgd-fr" },
                        { zob::batch_optimizer::CGD_PRP,      "cgd-prp" },
                        { zob::batch_optimizer::CGD_CD,       "cgd-cd" },
                        { zob::batch_optimizer::CGD_LS,       "cgd-ls" },
                        { zob::batch_optimizer::CGD_DY,       "cgd-dy" },
                        { zob::batch_optimizer::CGD_N,        "cgd-n" },
                        { zob::batch_optimizer::CGD_DYCD,     "cgd-dycd" },
                        { zob::batch_optimizer::CGD_DYHS,     "cgd-dyhs" }
                };
        }

        template <>
        inline std::map<zob::ls_initializer, std::string> enum_string<zob::ls_initializer>()
        {
                return
                {
                        { zob::ls_initializer::unit,          "init-unit" },
                        { zob::ls_initializer::quadratic,     "init-quadratic" },
                        { zob::ls_initializer::consistent,    "init-consistent" }
                };
        }

        template <>
        inline std::map<zob::ls_strategy, std::string> enum_string<zob::ls_strategy>()
        {
                return
                {
                        { zob::ls_strategy::backtrack_armijo,         "backtrack-Armijo" },
                        { zob::ls_strategy::backtrack_wolfe,          "backtrack-Wolfe" },
                        { zob::ls_strategy::backtrack_strong_wolfe,   "backtrack-strong-Wolfe" },
                        { zob::ls_strategy::interpolation,            "interp" },
                        { zob::ls_strategy::cg_descent,               "cgdescent" }
                };
        }
}


