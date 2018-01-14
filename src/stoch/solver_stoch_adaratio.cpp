#include "math/numeric.h"
#include "math/epsilon.h"
#include "math/momentum.h"
#include "solver_stoch_adaratio.h"

using namespace nano;

strings_t stoch_adaratio_t::configs() const
{
        strings_t configs;

        for (const auto alpha0 : make_scalars(1e-3, 1e-2, 1e-1, 1e+0))
        for (const auto momentum : make_scalars(0.10, 0.50, 0.90))
        for (const auto ratio0 : make_scalars(0.10, 0.50, 0.90, 0.95))
        for (const auto poly : make_scalars(1, 2, 3))
        {
                configs.push_back(json_writer_t().object(
                        "alpha0", alpha0, "momentum", momentum, "ratio0", ratio0, "poly", poly).str());
        }

        return configs;
}

json_reader_t& stoch_adaratio_t::config(json_reader_t& reader)
{
        return reader.object("alpha0", m_alpha0, "momentum", m_momentum, "ratio0", m_ratio0, "poly", m_poly);
}

json_writer_t& stoch_adaratio_t::config(json_writer_t& writer) const
{
        return writer.object("alpha0", m_alpha0, "momentum", m_momentum, "ratio0", m_ratio0, "poly", m_poly);
}

solver_state_t stoch_adaratio_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        // current learning rate
        scalar_t alpha = m_alpha0;

        // first-order momentum of the loss decrease ratio
        momentum1_t<scalar_t> ratio(m_momentum);

        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // descent direction
                cstate.d = -cstate.g;

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const solver_state_t& cstate, solver_state_t& sstate)
        {
                const auto prevf = sstate.f;
                sstate.update(function, cstate.x);
                const auto nextf = sstate.f;

                // update learning rate towards reaching the optimum function value decrease ratio (m_ratio0)
                const auto eps = epsilon0<scalar_t>();
                const auto ratiof = (eps + std::fabs(nextf)) / (eps + std::fabs(prevf));
                ratio.update(clamp(ratiof, scalar_t(0), scalar_t(1)));
                //todo: app --function regexp for benchmark_stoch & benchmark_batch
                //todo: this fails dramatically for the raynolds function
                //todo: double check that alpha or ratio don't grow too big
                alpha = alpha * std::pow(scalar_t(1) + ratio.value() - m_ratio0, m_poly);
        };

        return loop(param, function, x0, solver, snapshot);
}
