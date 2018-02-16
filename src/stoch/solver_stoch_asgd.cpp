#include "lrate.h"
#include "tensor/momentum.h"
#include "solver_stoch_asgd.h"

using namespace nano;

tuner_t stoch_asgd_t::configs() const
{
        tuner_t tuner;
        tuner.add_base10("alpha0", -4, 0);
        tuner.add_linear("decay", 0.5, 1.0);
        tuner.add_base10("tnorm", 0, 2);
        tuner.add_linear("momentum", 0.1, 0.9);
        return tuner;
}

json_reader_t& stoch_asgd_t::config(json_reader_t& reader)
{
        return reader.object("alpha0", m_alpha0, "decay", m_decay, "tnorm", m_tnorm, "momentum", m_momentum);
}

json_writer_t& stoch_asgd_t::config(json_writer_t& writer) const
{
        return writer.object("alpha0", m_alpha0, "decay", m_decay, "tnorm", m_tnorm, "momentum", m_momentum);
}

solver_state_t stoch_asgd_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        // learning rate schedule
        lrate_t lrate(m_alpha0, m_decay, m_tnorm);

        // average state
        momentum_t<vector_t> xavg(m_momentum, x0.size());

        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                xavg.update(cstate.x);
                cstate.d = -cstate.g;

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const solver_state_t&, solver_state_t& sstate)
        {
                sstate.update(function, xavg.value());
        };

        return loop(param, function, x0, solver, snapshot);
}
