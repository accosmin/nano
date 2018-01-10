#include "solver_stoch_ngd.h"

using namespace nano;

strings_t stoch_ngd_t::configs() const
{
        strings_t configs;

        for (const auto alpha0 : make_scalars(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0))
        {
                configs.push_back(json_writer_t().object(
                        "alpha0", alpha0).str());
        }

        return configs;
}

json_reader_t& stoch_ngd_t::config(json_reader_t& reader)
{
        return reader.object("alpha0", m_alpha0);
}

json_writer_t& stoch_ngd_t::config(json_writer_t& writer) const
{
        return writer.object("alpha0", m_alpha0);
}

solver_state_t stoch_ngd_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // learning rate
                const scalar_t alpha = m_alpha0;

                // descent direction
                const scalar_t norm = 1 / cstate.g.template lpNorm<2>();
                cstate.d = -cstate.g * norm;

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
