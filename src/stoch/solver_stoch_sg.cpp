#include "lrate.h"
#include "solver_stoch_sg.h"
#include "text/json_writer.h"

using namespace nano;

function_state_t stoch_sg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        return tune(this, param, function, x0, make_alpha0s(), make_decays());
}

function_state_t stoch_sg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
        const scalar_t alpha0, const scalar_t decay)
{
        // learning rate schedule
        lrate_t lrate(alpha0, decay);

        // assembly the solver
        const auto solver = [&] (function_state_t& cstate, const function_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                cstate.d = -cstate.g;

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const function_state_t& cstate, function_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return  loop(param, function, x0, solver, snapshot,
                json_writer_t().object("alpha0", alpha0, "decay", decay).get());
}
