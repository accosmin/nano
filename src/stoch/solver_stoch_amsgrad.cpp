#include "lrate.h"
#include "math/numeric.h"
#include "tensor/momentum.h"
#include "solver_stoch_amsgrad.h"

using namespace nano;

tuner_t stoch_amsgrad_t::configs() const
{
        tuner_t tuner;
        tuner.add_finite("alpha0", make_scalars(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e+0)).precision(3);
        tuner.add_finite("decay", make_scalars(0.0, 0.1, 0.2, 0.5, 0.9)).precision(1);
        return tuner;
}

json_reader_t& stoch_amsgrad_t::config(json_reader_t& reader)
{
        return reader.object("alpha0", m_alpha0, "decay", m_decay, "beta1", m_beta1, "beta2", m_beta2, "epsilon", m_epsilon);
}

json_writer_t& stoch_amsgrad_t::config(json_writer_t& writer) const
{
        return writer.object("alpha0", m_alpha0, "decay", m_decay, "beta1", m_beta1, "beta2", m_beta2, "epsilon", m_epsilon);
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