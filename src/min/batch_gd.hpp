#pragma once

#include "batch_params.hpp"
#include "linesearch_init.hpp"
#include "linesearch_strategy.hpp"

namespace min
{
        ///
        /// \brief gradient descent
        ///
        template
        <
                typename tproblem                       ///< optimization problem
        >
        struct batch_gd_t
        {
                typedef batch_params_t<tproblem>        param_t;
                typedef typename param_t::tscalar       tscalar;
                typedef typename param_t::tsize         tsize;
                typedef typename param_t::tvector       tvector;
                typedef typename param_t::tstate        tstate;
                typedef typename param_t::top_ulog      top_ulog;

                ///
                /// \brief constructor
                ///
                batch_gd_t(     tsize max_iterations,
                                tscalar epsilon,
                                ls_initializer lsinit,
                                ls_strategy lsstrat,
                                const top_ulog& ulog = top_ulog())
                        :       m_param(max_iterations, epsilon, lsinit, lsstrat, ulog)
                {
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const tproblem& problem, const tvector& x0) const
                {
                        assert(problem.size() == static_cast<tsize>(x0.size()));

                        tstate cstate(problem, x0);             // current state

                        // line-search initial step length
                        linesearch_init_t<tstate> ls_init(m_param.m_ls_initializer);

                        // line-search step
                        linesearch_strategy_t<tproblem> ls_step(m_param.m_ls_strategy, 1e-4, 0.1);

                        // iterate until convergence
                        for (tsize i = 0; i < m_param.m_max_iterations && m_param.ulog(cstate); i ++)
                        {
                                // check convergence
                                if (cstate.converged(m_param.m_epsilon))
                                {
                                        break;
                                }

                                // descent direction
                                cstate.d = -cstate.g;

                                // line-search
                                const tscalar t0 = ls_init(cstate);
                                if (!ls_step.update(problem, t0, cstate))
                                {
                                        break;
                                }
                        }

                        // OK
                        return cstate;
                }

                // attributes
                param_t         m_param;
        };
}

