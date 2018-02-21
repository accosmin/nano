#include "lrate.h"
#include "math/numeric.h"
#include "tensor/momentum.h"
#include "solver_stoch_amsgrad.h"

using namespace nano;

tuner_t stoch_amsgrad_t::configs() const
{
        tuner_t tuner;
        tuner.add_finite("alpha0", make_scalars(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e+0)).precision(3);
        tuner.add_finite("tnorm", make_scalars(1, 10, 100)).precision(0);
        return tuner;
}

json_reader_t& stoch_amsgrad_t::config(json_reader_t& reader)
{
        return reader.object("alpha0", m_alpha0, "decay", m_decay, "tnorm", m_tnorm, "beta1", m_beta1, "beta2", m_beta2);
}

json_writer_t& stoch_amsgrad_t::config(json_writer_t& writer) const
{
        return writer.object("alpha0", m_alpha0, "decay", m_decay, "tnorm", m_tnorm, "beta1", m_beta1, "beta2", m_beta2);
}

solver_state_t stoch_amsgrad_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        // learning rate schedule
        lrate_t lrate(m_alpha0, m_decay, m_tnorm);

        // first-order momentum of the gradient
        momentum_t<vector_t> m(m_beta1, x0.size());

        // second-order momentum of the gradient
        momentum_t<vector_t> v(m_beta2, x0.size());

        // maximum second-order momentum of the gradient
        vector_t vhat = vector_t::Zero(x0.size());

        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                m.update(cstate.g);
                v.update(cstate.g.array().square());
                vhat = vhat.array().max(v.value().array());

                cstate.d = -m.value().array() / vhat.array().sqrt();

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const solver_state_t& cstate, solver_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return loop(param, function, x0, solver, snapshot);
}
