#include "trainer_state.h"
#include "math/stats.hpp"
#include "text/align.hpp"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include <fstream>
#include <limits>
#include <cmath>

namespace zob
{
        trainer_state_t::trainer_state_t()
                :       trainer_state_t(
                        std::chrono::milliseconds(0),
                        size_t(0),
                        std::numeric_limits<scalar_t>::max(),
                        std::numeric_limits<scalar_t>::max(),
                        std::numeric_limits<scalar_t>::max(),
                        std::numeric_limits<scalar_t>::max(),
                        std::numeric_limits<scalar_t>::max(),
                        std::numeric_limits<scalar_t>::max())
        {
        }

        trainer_state_t::trainer_state_t(
                        std::chrono::milliseconds milis,
                        size_t epoch,
                        scalar_t tvalue,
                        scalar_t terror_avg,
                        scalar_t terror_var,
                        scalar_t vvalue,
                        scalar_t verror_avg,
                        scalar_t verror_var)
                :       m_milis(milis),
                        m_epoch(epoch),
                        m_tvalue(tvalue),
                        m_terror_avg(terror_avg),
                        m_terror_var(terror_var),
                        m_vvalue(vvalue),
                        m_verror_avg(verror_avg),
                        m_verror_var(verror_var)
        {
        }

        scalar_t convergence_speed(const trainer_states_t& states)
        {
                const auto op = [](const trainer_state_t& prv_state, const trainer_state_t& crt_state)
                {
                        assert(crt_state.m_tvalue >= scalar_t(0));
                        assert(prv_state.m_tvalue >= scalar_t(0));
                        assert(crt_state.m_milis >= prv_state.m_milis);

                        const scalar_t epsilon = zob::epsilon0<scalar_t>();
                        const auto ratio = (epsilon + crt_state.m_tvalue) / (epsilon + prv_state.m_tvalue);
                        const auto delta = 1 + crt_state.m_milis.count() - prv_state.m_milis.count();

                        // convergence speed ~ loss decrease ratio / second
                        return scalar_t(1000) / static_cast<scalar_t>(delta) * std::log(ratio);
                };

                zob::stats_t<scalar_t> speeds;
                for (size_t i = 0; i + 1 < states.size(); ++ i)
                {
                        speeds(op(states[i], states[i + 1]));
                }

                return speeds.avg();
        }

        bool operator<(const trainer_state_t& one, const trainer_state_t& two)
        {
                const scalar_t v1 = std::isfinite(one.m_verror_avg) ? one.m_verror_avg : std::numeric_limits<scalar_t>::max();
                const scalar_t v2 = std::isfinite(two.m_verror_avg) ? two.m_verror_avg : std::numeric_limits<scalar_t>::max();
                return v1 < v2;
        }

        bool save(const string_t& path, const trainer_states_t& states)
        {
                std::ofstream ofs(path.c_str(), std::ofstream::out);
                if (!ofs.is_open())
                {
                        return false;
                }

                const string_t delim = "\t";
                const size_t colsize = 24;

                // header
                ofs 
                << zob::align("train-loss", colsize) << delim
                << zob::align("train-error-average", colsize) << delim
                << zob::align("train-error-variance", colsize) << delim
                << zob::align("valid-loss", colsize) << delim
                << zob::align("valid-error-average", colsize) << delim
                << zob::align("valid-error-variance", colsize) << delim
                << zob::align("time-seconds", colsize) << delim
                << std::endl;

                // optimization states
                for (const trainer_state_t& state : states)
                {
                        ofs 
                        << zob::align(zob::to_string(state.m_tvalue), colsize) << delim
                        << zob::align(zob::to_string(state.m_terror_avg), colsize) << delim
                        << zob::align(zob::to_string(state.m_terror_var), colsize) << delim
                        << zob::align(zob::to_string(state.m_vvalue), colsize) << delim
                        << zob::align(zob::to_string(state.m_verror_avg), colsize) << delim
                        << zob::align(zob::to_string(state.m_verror_var), colsize) << delim
                        << zob::align(zob::to_string((state.m_milis.count() + 500) / 1000), colsize) << delim
                        << std::endl;
                }

                return ofs.good();
        }
}

