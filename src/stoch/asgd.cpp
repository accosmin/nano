#include "asgd.h"
#include "loop.h"
#include "lrate.h"
#include "math/average.h"
#include "text/to_params.h"

namespace nano
{
        stoch_asgd_t::stoch_asgd_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_asgd_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
        {
                return stoch_tune(this, param, function, x0, make_alpha0s(), make_decays());
        }

        state_t stoch_asgd_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay) const
        {
                assert(function.size() == x0.size());

                const auto config = to_params("alpha0", alpha0, "decay", decay);

                // learning rate schedule
                lrate_t lrate(alpha0, decay, param.m_epoch_size);

                // average
                average_vector_t<vector_t> xavg(x0.size());

                // current state
                state_t cstate = make_stoch_state(function, x0);

                // final state
                state_t fstate(function.size());

                // for each epoch ...
                for (size_t e = 0; e < param.m_max_epochs; ++ e)
                {
                        // for each iteration ...
                        for (size_t i = 0; i < param.m_epoch_size && cstate; ++ i)
                        {
                                // learning rate
                                const scalar_t alpha = lrate.get();

                                // descent direction
                                xavg.update(cstate.x);
                                cstate.d = -cstate.g;

                                // update solution
                                function.stoch_next();
                                cstate.stoch_update(function, alpha);
                        }

                        // check divergence
                        if (!cstate)
                        {
                                fstate.m_status = opt_status::failed;
                                break;
                        }

                        // check convergence (using the full gradient)
                        fstate.update(function, xavg.value());
                        if (fstate.converged(param.m_epsilon))
                        {
                                fstate.m_status = opt_status::converged;
                                param.tlog(fstate, config);
                                param.ulog(fstate, config);
                                break;
                        }

                        // log the current state & check the stopping criteria
                        param.tlog(fstate, config);
                        if (!param.ulog(fstate, config))
                        {
                                fstate.m_status = opt_status::stopped;
                                break;
                        }
                }

                // OK
                return fstate;
        }
}

