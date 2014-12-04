#pragma once

#include "stoch_params.hpp"
#include "decay.hpp"
#include <cassert>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief stochastic gradient (descent)
                ///
                /// NB: "Minimizing Finite Sums with the Stochastic Average Gradient"
                ///     - Mark Schmidth, Nicolas Le Roux, Francis Bach
                ///
                template
                <
                        decay_rate tbeta,               ///< learning rate's decay rate
                        typename tproblem               ///< optimization problem
                >
                struct stoch_sg : public stoch_params<tproblem>
                {
                        typedef stoch_params<tproblem>          base_t;

                        typedef typename base_t::tscalar        tscalar;
                        typedef typename base_t::tsize          tsize;
                        typedef typename base_t::tvector        tvector;
                        typedef typename base_t::tstate         tstate;
                        typedef typename base_t::tulog          tulog;

                        ///
                        /// \brief constructor
                        ///
                        stoch_sg(       tsize epochs,
                                        tsize epoch_size,
                                        tscalar alpha0,
                                        const tulog& ulog = tulog())
                                :       base_t(epochs, epoch_size, alpha0, ulog)
                        {
                        }

                        ///
                        /// \brief minimize starting from the initial guess x0
                        ///
                        tstate operator()(const tproblem& problem, const tvector& x0) const
                        {
                                assert(problem.size() == static_cast<tsize>(x0.size()));

                                tstate cstate(problem, x0);             // current state

                                for (tsize e = 0, k = 0; e < base_t::m_epochs; e ++)
                                {
                                        for (tsize i = 0; i < base_t::m_epoch_size; i ++)
                                        {
                                                // learning rate
                                                const tscalar alpha = optimize::decay(base_t::m_alpha0, k ++, tbeta);

                                                // descent direction
                                                cstate.d = -cstate.g;

                                                // update solution
                                                cstate.update(problem, alpha);
                                        }

                                        base_t::ulog(cstate);
                                }

                                return cstate;
                        }
                };
        }
}

