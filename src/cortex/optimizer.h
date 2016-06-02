#pragma once

#include "scalar.h"
#include "math/problem.hpp"
#include "math/batch_types.h"
#include "math/stoch_types.h"
#include "math/lsearch_types.h"
#include "text/enum_string.hpp"

namespace nano
{
        template <>
        inline std::map<nano::stoch_optimizer, std::string> enum_string<nano::stoch_optimizer>()
        {
                return
                {
                        { nano::stoch_optimizer::SG,           "sg" },
                        { nano::stoch_optimizer::NGD,          "ngd" },
                        { nano::stoch_optimizer::SGM,          "sgm" },
                        { nano::stoch_optimizer::AG,           "ag" },
                        { nano::stoch_optimizer::AGFR,         "agfr" },
                        { nano::stoch_optimizer::AGGR,         "aggr" },
                        { nano::stoch_optimizer::ADAGRAD,      "adagrad" },
                        { nano::stoch_optimizer::ADADELTA,     "adadelta" },
                        { nano::stoch_optimizer::ADAM,         "adam" }
                };
        }

        template <>
        inline std::map<nano::batch_optimizer, std::string> enum_string<nano::batch_optimizer>()
        {
                return
                {
                        { nano::batch_optimizer::GD,           "gd" },
                        { nano::batch_optimizer::CGD,          "cgd" },
                        { nano::batch_optimizer::LBFGS,        "lbfgs" },
                        { nano::batch_optimizer::CGD_HS,       "cgd-hs" },
                        { nano::batch_optimizer::CGD_FR,       "cgd-fr" },
                        { nano::batch_optimizer::CGD_PRP,      "cgd-prp" },
                        { nano::batch_optimizer::CGD_CD,       "cgd-cd" },
                        { nano::batch_optimizer::CGD_LS,       "cgd-ls" },
                        { nano::batch_optimizer::CGD_DY,       "cgd-dy" },
                        { nano::batch_optimizer::CGD_N,        "cgd-n" },
                        { nano::batch_optimizer::CGD_DYCD,     "cgd-dycd" },
                        { nano::batch_optimizer::CGD_DYHS,     "cgd-dyhs" }
                };
        }

        template <>
        inline std::map<nano::ls_initializer, std::string> enum_string<nano::ls_initializer>()
        {
                return
                {
                        { nano::ls_initializer::unit,          "init-unit" },
                        { nano::ls_initializer::quadratic,     "init-quadratic" },
                        { nano::ls_initializer::consistent,    "init-consistent" }
                };
        }

        template <>
        inline std::map<nano::ls_strategy, std::string> enum_string<nano::ls_strategy>()
        {
                return
                {
                        { nano::ls_strategy::backtrack_armijo,         "backtrack-Armijo" },
                        { nano::ls_strategy::backtrack_wolfe,          "backtrack-Wolfe" },
                        { nano::ls_strategy::backtrack_strong_wolfe,   "backtrack-strong-Wolfe" },
                        { nano::ls_strategy::interpolation,            "interp" },
                        { nano::ls_strategy::cg_descent,               "cgdescent" }
                };
        }
}


