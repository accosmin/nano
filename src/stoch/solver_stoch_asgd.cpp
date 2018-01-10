#include "lrate.h"
#include "tensor/momentum.h"
#include "solver_stoch_asgd.h"

using namespace nano;

strings_t stoch_asgd_t::configs() const
{
        strings_t configs;

        for (const auto alpha0 : make_scalars(1e-3, 1e-2, 1e-1))
        for (const auto decay : make_scalars(0.00, 0.50, 0.75, 1.00))
        for (const auto momentum : make_scalars(0.10, 0.50, 0.90))
        {
                configs.push_back(json_writer_t().object(
                        "alpha0", alpha0, "decay", decay, "momentum", momentum).str());
        }

        return configs;
}

json_reader_t& stoch_asgd_t::config(json_reader_t& reader)
{
        return reader.object("alpha0", m_alpha0, "decay", m_decay, "momentum", m_momentum);
}

json_writer_t& stoch_asgd_t::config(json_writer_t& writer) const
{
        return writer.object("alpha0", m_alpha0, "decay", m_decay, "momentum", m_momentum);
}

solver_state_t stoch_asgd_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        // learning rate schedule
        lrate_t lrate(m_alpha0, m_decay);

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
