#pragma once

#include "stoch_params.hpp"
#include "../math/average_vector.hpp"
#include <cassert>

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief stochastic AdaDelta,
                ///     see "ADADELTA: An Adaptive Learning Rate Method", by Matthew D. Zeiler
                ///
                template
                <
                        typename tproblem               ///< optimization problem
                >
                struct stoch_adadelta_t : public stoch_params_t<tproblem>
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
                        stoch_adadelta_t(tsize epochs,
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

                                // running-weighted-averaged-per-dimension-squared gradient
                                average_vector_t<tscalar, tvector> gavg(x0.size());

                                // running-weighted-averaged-per-dimension-squared step updates
                                average_vector_t<tscalar, tvector> davg(x0.size());

                                for (tsize e = 0, k = 1; e < base_t::m_epochs; e ++)
                                {
                                        for (tsize i = 0; i < base_t::m_epoch_size; i ++, k ++)
                                        {
                                                // descent direction
                                                gavg.update(cstate.g.array().square(), base_t::weight(k));

                                                cstate.d = -cstate.g.array() *
                                                           (base_t::m_epsilon + davg.value().array()).sqrt() /
                                                           (base_t::m_epsilon + gavg.value().array()).sqrt();

                                                davg.update(cstate.d.array().square(), base_t::weight(k));

                                                // update solution
                                                cstate.update(problem, tscalar(1));
                                        }

                                        base_t::ulog(cstate);
                                }

                                return cstate;
                        }
                };
        }
}

