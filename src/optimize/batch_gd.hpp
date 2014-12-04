#pragma once

#include "batch_params.hpp"
#include "ls_armijo.hpp"
#include <cassert>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief gradient descent
                ///
                template
                <
                        typename tproblem                       ///< optimization problem
                >
                struct batch_gd : public batch_params<tproblem>
                {
                        typedef batch_params<tproblem>          base_t;

                        typedef typename base_t::tscalar        tscalar;
                        typedef typename base_t::tsize          tsize;
                        typedef typename base_t::tvector        tvector;
                        typedef typename base_t::tstate         tstate;
                        typedef typename base_t::twlog          twlog;
                        typedef typename base_t::telog          telog;
                        typedef typename base_t::tulog          tulog;

                        ///
                        /// \brief constructor
                        ///
                        batch_gd(       tsize max_iterations,
                                        tscalar epsilon,
                                        const twlog& wlog = twlog(),
                                        const telog& elog = telog(),
                                        const tulog& ulog = tulog())
                                :       base_t(max_iterations, epsilon, wlog, elog, ulog)
                        {
                        }

                        ///
                        /// \brief minimize starting from the initial guess x0
                        ///
                        tstate operator()(const tproblem& problem, const tvector& x0) const
                        {
                                assert(problem.size() == static_cast<tsize>(x0.size()));

                                tstate cstate(problem, x0);     // current state

                                // iterate until convergence
                                for (tsize i = 0; i < base_t::m_max_iterations; i ++)
                                {
                                        base_t::ulog(cstate);

                                        // check convergence
                                        if (cstate.converged(base_t::m_epsilon))
                                        {
                                                break;
                                        }

                                        // descent direction
                                        cstate.d = -cstate.g;

                                        // update solution
                                        const tscalar t = ls_armijo(problem, cstate, base_t::m_wlog);
                                        if (t < std::numeric_limits<tscalar>::epsilon())
                                        {
                                                base_t::elog("line-search failed for GD!");
                                                break;
                                        }
                                        cstate.update(problem, t);
                                }

                                return cstate;
                        }
                };
        }
}

