#include "loop.h"
#include "tensor/momentum.h"
#include "solver_stoch_adadelta.h"

using namespace nano;

stoch_adadelta_t::stoch_adadelta_t(const string_t& params) :
        stoch_solver_t(params)
{
}

function_state_t stoch_adadelta_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        return stoch_tune(this, param, function, x0, make_momenta(), make_epsilons());
}

function_state_t stoch_adadelta_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
        const scalar_t momentum, const scalar_t epsilon)
{
        // second-order momentum of the gradient
        momentum_t<vector_t> gavg(momentum, x0.size());

        // second-order momentum of the step updates
        momentum_t<vector_t> davg(momentum, x0.size());

        // assembly the solver
        const auto solver = [&] (function_state_t& cstate, const function_state_t&)
        {
                // descent direction
                gavg.update(cstate.g.array().square());

                cstate.d = -cstate.g.array() *
                           (epsilon + davg.value().array().sqrt()) /
                           (epsilon + gavg.value().array().sqrt());

                davg.update(cstate.d.array().square());

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, 1);
        };

        const auto snapshot = [&] (const function_state_t& cstate, function_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return  stoch_loop(param, function, x0, solver, snapshot,
                to_params("momentum", momentum, "epsilon", epsilon));
}
