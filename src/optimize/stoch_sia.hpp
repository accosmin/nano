#pragma once

#include "stoch_params.hpp"
#include "average.hpp"
#include <cassert>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief stochastic iterative average gradient (descent)
                ///
                /// NB: "Minimizing Finite Sums with the Stochastic Average Gradient"
                ///     - Mark Schmidth, Nicolas Le Roux, Francis Bach
                ///
                template
                <
                        typename tproblem               ///< optimization problem
                >
                struct stoch_sia : public stoch_params_t<tproblem>
                {
                        typedef stoch_params_t<tproblem>        base_t;

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
                        stoch_sia(      tsize epochs,
                                        tsize epoch_size,
                                        tscalar alpha0,
                                        tscalar decay,
                                        const twlog& wlog = twlog(),
                                        const telog& elog = telog(),
                                        const tulog& ulog = tulog())
                                :       base_t(epochs, epoch_size, alpha0, decay, wlog, elog, ulog)
                        {
                        }

                        ///
                        /// \brief minimize starting from the initial guess x0
                        ///
                        tstate operator()(const tproblem& problem, const tvector& x0) const
                        {
                                assert(problem.size() == static_cast<tsize>(x0.size()));

                                // current state
                                tstate cstate(problem, x0);

                                // running-weighted-averaged parameters
                                average_vector<tscalar, tvector> xavg(x0.size());

                                for (tsize e = 0, k = 0; e < base_t::m_epochs; e ++)
                                {
                                        for (tsize i = 0; i < base_t::m_epoch_size; i ++)
                                        {
                                                // learning rate
                                                const tscalar alpha = base_t::alpha(k ++);

                                                // descent direction
                                                cstate.d = -cstate.g;

                                                // update solution
                                                cstate.update(problem, alpha);

                                                xavg.update(cstate.x, tscalar(1) / alpha);
                                        }

                                        const tvector cx = cstate.x;
                                        cstate.x = xavg.value();        // NB: to correctly log the current parameters!
                                        base_t::ulog(cstate);
                                        cstate.x = cx;                  // revert it
                                }

                                // OK, setup the average parameter as the final result
                                cstate.x = xavg.value();
                                return cstate;
                        }
                };
        }
}

