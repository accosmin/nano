#pragma once

#include "tensor.h"
#include "min/batch.h"
#include "min/stoch.h"
#include "min/problem.hpp"
#include "min/linesearch.h"
#include "text/enum_string.hpp"

namespace ncv
{
        typedef ::min::problem_t<scalar_t>              opt_problem_t;

        typedef typename opt_problem_t::tscalar         opt_scalar_t;
        typedef typename opt_problem_t::tvector         opt_vector_t;
        typedef typename opt_problem_t::tstate          opt_state_t;
        typedef typename opt_problem_t::tsize           opt_size_t;

        typedef typename opt_problem_t::top_size        opt_opsize_t;
        typedef typename opt_problem_t::top_fval        opt_opfval_t;
        typedef typename opt_problem_t::top_grad        opt_opgrad_t;
        typedef typename opt_problem_t::top_ulog        opt_opulog_t;

        // string cast for enumerations
        namespace text
        {
                template <>
                inline std::map<::min::stoch_optimizer, std::string> enum_string<::min::stoch_optimizer>()
                {
                        return
                        {
                                { ::min::stoch_optimizer::SG,           "sg" },
                                { ::min::stoch_optimizer::SGA,          "sga" },
                                { ::min::stoch_optimizer::SIA,          "sia" },
                                { ::min::stoch_optimizer::AG,           "ag" },
                                { ::min::stoch_optimizer::AGGR,         "aggr" },
                                { ::min::stoch_optimizer::ADAGRAD,      "adagrad" },
                                { ::min::stoch_optimizer::ADADELTA,     "adadelta" }
                        };
                }

                template <>
                inline std::map<::min::batch_optimizer, std::string> enum_string<::min::batch_optimizer>()
                {
                        return
                        {
                                { ::min::batch_optimizer::GD,           "gd" },
                                { ::min::batch_optimizer::CGD,          "cgd" },
                                { ::min::batch_optimizer::LBFGS,        "lbfgs" },
                                { ::min::batch_optimizer::CGD_HS,       "cgd-hs" },
                                { ::min::batch_optimizer::CGD_FR,       "cgd-fr" },
                                { ::min::batch_optimizer::CGD_PRP,      "cgd-prp" },
                                { ::min::batch_optimizer::CGD_CD,       "cgd-cd" },
                                { ::min::batch_optimizer::CGD_LS,       "cgd-ls" },
                                { ::min::batch_optimizer::CGD_DY,       "cgd-dy" },
                                { ::min::batch_optimizer::CGD_N,        "cgd-n" },
                                { ::min::batch_optimizer::CGD_DYCD,     "cgd-dycd" },
                                { ::min::batch_optimizer::CGD_DYHS,     "cgd-dyhs" }
                        };
                }

                template <>
                inline std::map<::min::ls_initializer, std::string> enum_string<::min::ls_initializer>()
                {
                        return
                        {
                                { ::min::ls_initializer::unit,          "init-unit" },
                                { ::min::ls_initializer::quadratic,     "init-quadratic" },
                                { ::min::ls_initializer::consistent,    "init-consistent" }
                        };
                }

                template <>
                inline std::map<::min::ls_strategy, std::string> enum_string<::min::ls_strategy>()
                {
                        return
                        {
                                { ::min::ls_strategy::backtrack_armijo,         "backtrack-Armijo" },
                                { ::min::ls_strategy::backtrack_wolfe,          "backtrack-Wolfe" },
                                { ::min::ls_strategy::backtrack_strong_wolfe,   "backtrack-strong-Wolfe" },
                                { ::min::ls_strategy::interpolation,            "interp" },
                                { ::min::ls_strategy::cg_descent,               "cgdescent" }
                        };
                }

                template <>
                inline std::map<::min::status, std::string> enum_string<::min::status>()
                {
                        return
                        {
                                { ::min::status::converged,                       "converged" },
                                { ::min::status::max_iterations,                  "maximum iterations" },
                                { ::min::status::ls_failed_invalid_step,          "linesearch failed (invalid step)" },
                                { ::min::status::ls_failed_not_decreasing_step,   "linesearch failed (not decreasing step)" },
                                { ::min::status::ls_failed_invalid_initial_step,  "linesearch failed (invalid initial step)" },
                                { ::min::status::ls_failed_not_descent,           "linesearch failed (not descent direction)" }
                        };
                }
        }
}


