#include "lrate.h"
#include "math/numeric.h"
#include "tensor/momentum.h"
#include "solver_stoch_amsgrad.h"

using namespace nano;

tuner_t stoch_amsgrad_t::configs() const
{
        tuner_t tuner;
        tuner.add("alpha0", make_pow10_scalars(0, -3, -1)).precision(3);
        tuner.add("decay", make_scalars(0.1, 0.2, 0.5, 0.9)).precision(1);
        tuner.add("epsilon", make_pow10_scalars(0, -7, -2)).precision(7);
        tuner.add("beta1", make_scalars(0.90, 0.95, 0.99)).precision(2);
        tuner.add("beta2", make_scalars(0.990, 0.999)).precision(3);
        return tuner;
}

void stoch_amsgrad_t::from_json(const json_t& json)
{
        nano::from_json(json, "alpha0", m_alpha0, "decay", m_decay, "beta1", m_beta1, "beta2", m_beta2, "epsilon", m_epsilon);
}

void stoch_amsgrad_t::to_json(json_t& json) const
{
        nano::to_json(json, "alpha0", m_alpha0, "decay", m_decay, "beta1", m_beta1, "beta2", m_beta2, "epsilon", m_epsilon);
}

solver_state_t stoch_amsgrad_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        lrate_t lrate(m_alpha0, m_decay);

        momentum_t<vector_t> m(m_beta1, x0.size());
        momentum_t<vector_t> v(m_beta2, x0.size());
        vector_t vhat = vector_t::Zero(x0.size());

        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // descent direction
                m.update(cstate.g);
                v.update(cstate.g.array().square());
                vhat = vhat.array().max(v.value().array());

                cstate.d = -m.value().array() / (m_epsilon + vhat.array().sqrt());

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, lrate.get());
        };

        const auto snapshot = [&] (const solver_state_t& cstate, solver_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return loop(param, function, x0, solver, snapshot);
}
