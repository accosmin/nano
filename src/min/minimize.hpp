#pragma once

#include "min/problem.hpp"
#include "min/linesearch.h"

#include "min/batch.h"
#include "min/batch/gd.hpp"
#include "min/batch/cgd.hpp"
#include "min/batch/lbfgs.hpp"

#include "min/stoch.h"
#include "min/stoch/ag.hpp"
#include "min/stoch/sg.hpp"
#include "min/stoch/sga.hpp"
#include "min/stoch/sia.hpp"
#include "min/stoch/adagrad.hpp"
#include "min/stoch/adadelta.hpp"

namespace min
{
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
                typename topulog = typename tproblem::top_ulog
        >
        tstate minimize(
                const tproblem& problem,
                const topulog& fn_ulog,
                const tvector& x0,
                batch_optimizer, std::size_t iterations, tscalar epsilon,
                std::size_t history_size = 6);

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
                typename topulog = typename tproblem::top_ulog
        >
        tstate minimize(
                const tproblem& problem,
                const topulog& fn_ulog,
                const tvector& x0,
                batch_optimizer, std::size_t iterations, tscalar epsilon,
                ls_initializer,
                ls_strategy,
                std::size_t history_size = 6);

        ///
        /// \brief stochastic optimization
        ///
        template
        <
                typename tscalar,
                typename tproblem = problem_t<tscalar>,
                typename tsize = typename tproblem::tsize,
                typename tstate = typename tproblem::tstate,
                typename tvector = typename tproblem::tvector,
                typename topulog = typename tproblem::top_ulog
        >
        tstate minimize(
                const tproblem& problem,
                const topulog& fn_ulog,
                const tvector& x0,
                stoch_optimizer, std::size_t epochs, std::size_t epoch_size, tscalar alpha0, tscalar decay = 0.50);

        template
        <
                typename tscalar,
                typename tproblem,
                typename tsize,
                typename tstate,
                typename tvector,
                typename topulog
        >
        tstate minimize(
                const tproblem& problem,
                const topulog& fn_ulog,
                const tvector& x0,
                batch_optimizer optimizer, std::size_t iterations, tscalar epsilon, std::size_t history_size)
        {
                switch (optimizer)
                {
                case batch_optimizer::LBFGS:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        ls_initializer::unit, ls_strategy::interpolation,
                                        history_size);

                case batch_optimizer::CGD:
                        return minimize(problem, fn_ulog, x0, batch_optimizer::CGD_PRP, iterations, epsilon,
                                        history_size);

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

        template
        <
                typename tscalar,
                typename tproblem,
                typename tsize,
                typename tstate,
                typename tvector,
                typename topulog
        >
        tstate minimize(
                const tproblem& problem,
                const topulog& fn_ulog,
                const tvector& x0,
                batch_optimizer optimizer, std::size_t iterations, tscalar epsilon,
                ls_initializer lsinit, ls_strategy lsstrat,
                std::size_t history_size)
        {
                switch (optimizer)
                {
//                case batch_optimizer::libLBFGS:
//                        return liblbfgs::minimize(problem, x0, iterations, epsilon, history_size);

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

        template
        <
                typename tscalar,
                typename tproblem,
                typename tsize,
                typename tstate,
                typename tvector,
                typename topulog
        >
        tstate minimize(
                const tproblem& problem,
                const topulog& fn_ulog,
                const tvector& x0,
                stoch_optimizer optimizer, std::size_t epochs, std::size_t epoch_size, tscalar alpha0, tscalar decay)
        {
                switch (optimizer)
                {
                case stoch_optimizer::SGA:
                        return  stoch_sga_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case stoch_optimizer::SIA:
                        return  stoch_sia_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case stoch_optimizer::AG:
                        return  stoch_ag_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case stoch_optimizer::AGGR:
                        return  stoch_aggr_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case stoch_optimizer::ADAGRAD:
                        return  stoch_adagrad_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case stoch_optimizer::ADADELTA:
                        return  stoch_adadelta_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case stoch_optimizer::SG:
                default:
                        return  stoch_sg_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);
                }
        }
}

