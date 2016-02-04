#pragma once

#include "ls_init.hpp"
#include "cgd_steps.hpp"
#include "batch_loop.hpp"
#include "ls_strategy.hpp"

namespace math
{
        ///
        /// \brief conjugate gradient descent
        ///
        template
        <
                typename tcgd_update,                   ///< CGD step update
                typename tproblem                       ///< optimization problem
        >
        struct batch_cgd_t
        {
                using param_t = batch_params_t<tproblem>;
                using tstate = typename param_t::tstate;
                using tscalar = typename param_t::tscalar;
                using tvector = typename param_t::tvector;
                using topulog = typename param_t::topulog;

                ///
                /// \brief constructor
                ///
                explicit batch_cgd_t(const param_t& param) : m_param(param)
                {
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const tproblem& problem, const tvector& x0) const
                {
                        assert(problem.size() == x0.size());

                        tstate istate(problem, x0);     // initial state
                        tstate pstate = istate;         // previous state

                        // line-search initial step length
                        ls_init_t<tstate> ls_init(m_param.m_ls_initializer);

                        // line-search step
                        ls_strategy_t<tproblem> ls_step(m_param.m_ls_strategy, 1e-4, 0.1);

                        // CGD direction strategy
                        const tcgd_update op_update;

                        const auto op = [&] (tstate& cstate, const std::size_t i)
                        {
                                // descent direction
                                if (i == 0)
                                {
                                        cstate.d = -cstate.g;
                                }
                                else
                                {
                                        const tscalar beta = op_update(pstate, cstate);
                                        cstate.d = -cstate.g + beta * pstate.d;
                                }

                                if (cstate.d.dot(cstate.g) > tscalar(0))
                                {
                                        // reset to gradient descent, if not a descent direction!
                                        cstate.d = -cstate.g;
                                }

                                // line-search
                                pstate = cstate;

                                const tscalar t0 = ls_init(cstate);
                                return ls_step(problem, t0, cstate);
                        };

                        // OK, assembly the optimizer
                        return batch_loop(m_param, istate, op);
                }

                // attributes
                param_t         m_param;
        };

        // create various CGD algorithms
        template <typename tproblem>
        using batch_cgd_hs_t = batch_cgd_t<cgd_step_HS<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_fr_t = batch_cgd_t<cgd_step_FR<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_prp_t = batch_cgd_t<cgd_step_PRP<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_cd_t = batch_cgd_t<cgd_step_CD<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_ls_t = batch_cgd_t<cgd_step_LS<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_dy_t = batch_cgd_t<cgd_step_DY<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_n_t = batch_cgd_t<cgd_step_N<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_dycd_t = batch_cgd_t<cgd_step_DYCD<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_dyhs_t = batch_cgd_t<cgd_step_DYHS<typename tproblem::tstate>, tproblem>;
}

