#include "lrate.h"
#include "text/json_writer.h"
#include "solver_stoch_svrg.h"

using namespace nano;

solver_state_t stoch_svrg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        return tune(this, param, function, x0, make_alpha0s(), make_decays());
}

solver_state_t stoch_svrg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
        const scalar_t alpha0, const scalar_t decay)
{
        // learning rate schedule
        lrate_t lrate(alpha0, decay);

        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t& sstate)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                function.stoch_eval(sstate.x, &cstate.d);// NB: reuse descent direction to store snapshot gradient!
                cstate.d.noalias() = - cstate.g + cstate.d - sstate.g;

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const solver_state_t& cstate, solver_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return  loop(param, function, x0, solver, snapshot,
                json_writer_t().object("alpha0", alpha0, "decay", decay).str());
}
