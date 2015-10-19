#pragma once

#include "problem.hpp"
#include "lsearch_types.h"

#include "batch_types.h"
#include "batch/gd.hpp"
#include "batch/cgd.hpp"
#include "batch/lbfgs.hpp"

namespace min
{
        ///
        /// \brief batch optimization (can detail the line-search parameters)
        ///
        template
        <
                typename tscalar,
                typename tproblem = problem_t<tscalar>,
                typename tsize = typename tproblem::tsize,
                typename tstate = typename tproblem::tstate,
                typename tvector = typename tproblem::tvector,
                typename topulog = typename tproblem::topulog
        >
        auto minimize(
                const tproblem& problem,
                const topulog& fn_ulog,
                const tvector& x0,
                batch_optimizer optimizer, std::size_t iterations, tscalar epsilon,
                ls_initializer lsinit,
                ls_strategy lsstrat,
                std::size_t history_size = 6)
        {
                switch (optimizer)
                {
                case batch_optimizer::LBFGS:
                        return  batch_lbfgs_t<tproblem>
                                (iterations, epsilon, lsinit, lsstrat, history_size, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD:
                        return  batch_cgd_prp_t<tproblem>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_CD:
                        return  batch_cgd_cd_t<tproblem>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_DY:
                        return  batch_cgd_dy_t<tproblem>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_FR:
                        return  batch_cgd_fr_t<tproblem>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_HS:
                        return  batch_cgd_hs_t<tproblem>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_LS:
                        return  batch_cgd_ls_t<tproblem>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_N:
                        return  batch_cgd_n_t<tproblem>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_PRP:
                        return  batch_cgd_prp_t<tproblem>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_DYCD:
                        return  batch_cgd_dycd_t<tproblem>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_DYHS:
                        return  batch_cgd_dyhs_t<tproblem>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case batch_optimizer::GD:
                default:
                        return  batch_gd_t<tproblem>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);
                }
        }

        ///
        /// \brief batch optimization
        ///
        template
        <
                typename tscalar,
                typename tproblem = problem_t<tscalar>,
                typename tsize = typename tproblem::tsize,
                typename tstate = typename tproblem::tstate,
                typename tvector = typename tproblem::tvector,
                typename topulog = typename tproblem::topulog
        >
        auto minimize(
                const tproblem& problem,
                const topulog& fn_ulog,
                const tvector& x0,
                batch_optimizer optimizer, std::size_t iterations, tscalar epsilon,
                std::size_t history_size = 6)
        {
                switch (optimizer)
                {
                case batch_optimizer::LBFGS:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        ls_initializer::unit, ls_strategy::interpolation,
                                        history_size);

                case batch_optimizer::CGD: // fall through!
                case batch_optimizer::CGD_PRP:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        ls_initializer::quadratic, ls_strategy::interpolation,
                                        history_size);

                case batch_optimizer::CGD_CD:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        ls_initializer::unit, ls_strategy::interpolation,
                                        history_size);

                case batch_optimizer::CGD_DY:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        ls_initializer::quadratic, ls_strategy::backtrack_wolfe,
                                        history_size);

                case batch_optimizer::CGD_FR:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        ls_initializer::quadratic, ls_strategy::backtrack_armijo,
                                        history_size);

                case batch_optimizer::CGD_HS:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        ls_initializer::quadratic, ls_strategy::backtrack_wolfe,
                                        history_size);

                case batch_optimizer::CGD_LS:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        ls_initializer::quadratic, ls_strategy::interpolation,
                                        history_size);

                case batch_optimizer::CGD_N:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        ls_initializer::quadratic, ls_strategy::interpolation,
                                        history_size);

                case batch_optimizer::CGD_DYCD:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        ls_initializer::unit, ls_strategy::interpolation,
                                        history_size);

                case batch_optimizer::CGD_DYHS:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        ls_initializer::quadratic, ls_strategy::interpolation,
                                        history_size);

                case batch_optimizer::GD:
                default:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        ls_initializer::quadratic, ls_strategy::backtrack_wolfe,
                                        history_size);
                }
        }
}

