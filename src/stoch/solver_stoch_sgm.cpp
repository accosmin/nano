#include "lrate.h"
#include "tensor/momentum.h"
#include "solver_stoch_sgm.h"

using namespace nano;

tuner_t stoch_sgm_t::configs() const
{
        tuner_t tuner;
        tuner.add_finite("alpha0", make_scalars(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e+0)).precision(3);
        tuner.add_finite("tnorm", make_scalars(1, 10, 100)).precision(0);
        tuner.add_finite("momentum", make_scalars(0.10, 0.20, 0.50, 0.90)).precision(2);
        return tuner;
}

json_reader_t& stoch_sgm_t::config(json_reader_t& reader)
{
        return reader.object("alpha0", m_alpha0, "decay", m_decay, "tnorm", m_tnorm, "momentum", m_momentum);
}

json_writer_t& stoch_sgm_t::config(json_writer_t& writer) const
{
        return writer.object("alpha0", m_alpha0, "decay", m_decay, "tnorm", m_tnorm, "momentum", m_momentum);
}

solver_state_t stoch_sgm_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        // learning rate schedule
        lrate_t lrate(m_alpha0, m_decay, m_tnorm);

        // first-order momentum of the gradient
        momentum_t<vector_t> gsum1(m_momentum, x0.size());

        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                gsum1.update(cstate.g);

                cstate.d = -gsum1.value();

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
