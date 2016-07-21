#include "trainer_state.h"
#include "math/stats.hpp"
#include "text/align.hpp"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include <fstream>
#include <limits>
#include <cmath>

namespace nano
{
        trainer_measurement_t::trainer_measurement_t() : trainer_measurement_t(
                std::numeric_limits<scalar_t>::max(),
                std::numeric_limits<scalar_t>::max(),
                std::numeric_limits<scalar_t>::max())
        {
        }

        trainer_measurement_t::trainer_measurement_t(
                const scalar_t value, const scalar_t error_avg, const scalar_t error_var) :
                m_value(value),
                m_error_avg(error_avg),
                m_error_var(error_var)
        {
        }

        bool operator<(const trainer_measurement_t& one, const trainer_measurement_t& two)
        {
                const auto v1 = std::isfinite(one.m_error_avg) ? one.m_error_avg : std::numeric_limits<scalar_t>::max();
                const auto v2 = std::isfinite(two.m_error_avg) ? two.m_error_avg : std::numeric_limits<scalar_t>::max();
                return v1 < v2;
        }

        trainer_state_t::trainer_state_t() :
                m_milis(0),
                m_epoch(0)
        {
        }

        trainer_state_t::trainer_state_t(
                const milliseconds_t milis,
                const size_t epoch,
                const trainer_measurement_t& train,
                const trainer_measurement_t& valid,
                const trainer_measurement_t& test) :
                m_milis(milis),
                m_epoch(epoch),
                m_train(train),
                m_valid(valid),
                m_test(test)
        {
        }

        scalar_t convergence_speed(const trainer_states_t& states)
        {
                const auto op = [](const trainer_state_t& prv, const trainer_state_t& crt)
                {
                        assert(crt.m_train.m_value >= scalar_t(0));
                        assert(prv.m_train.m_value >= scalar_t(0));
                        assert(crt.m_milis >= prv.m_milis);

                        const scalar_t epsilon = nano::epsilon0<scalar_t>();
                        const auto ratio = (epsilon + crt.m_train.m_value) / (epsilon + prv.m_train.m_value);
                        const auto delta = 1 + crt.m_milis.count() - prv.m_milis.count();

                        // convergence speed ~ loss decrease ratio / second
                        return scalar_t(1000) / static_cast<scalar_t>(delta) * std::log(ratio);
                };

                nano::stats_t<scalar_t> speeds;
                for (size_t i = 0; i + 1 < states.size(); ++ i)
                {
                        speeds(op(states[i], states[i + 1]));
                }

                return static_cast<scalar_t>(speeds.avg());
        }

        bool operator<(const trainer_state_t& one, const trainer_state_t& two)
        {
                // compare (aka tune) on the validation dataset!
                return one.m_valid < two.m_valid;
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
                << nano::align("train-loss", colsize) << delim
                << nano::align("train-error-average", colsize) << delim
                << nano::align("train-error-variance", colsize) << delim
                << nano::align("valid-loss", colsize) << delim
                << nano::align("valid-error-average", colsize) << delim
                << nano::align("valid-error-variance", colsize) << delim
                << nano::align("test-loss", colsize) << delim
                << nano::align("test-error-average", colsize) << delim
                << nano::align("test-error-variance", colsize) << delim
                << nano::align("time-seconds", colsize) << delim
                << std::endl;

                // optimization states
                for (const trainer_state_t& state : states)
                {
                        ofs
                        << nano::align(nano::to_string(state.m_train.m_value), colsize) << delim
                        << nano::align(nano::to_string(state.m_train.m_error_avg), colsize) << delim
                        << nano::align(nano::to_string(state.m_train.m_error_var), colsize) << delim
                        << nano::align(nano::to_string(state.m_valid.m_value), colsize) << delim
                        << nano::align(nano::to_string(state.m_valid.m_error_avg), colsize) << delim
                        << nano::align(nano::to_string(state.m_valid.m_error_var), colsize) << delim
                        << nano::align(nano::to_string(state.m_test.m_value), colsize) << delim
                        << nano::align(nano::to_string(state.m_test.m_error_avg), colsize) << delim
                        << nano::align(nano::to_string(state.m_test.m_error_var), colsize) << delim
                        << nano::align(nano::to_string((state.m_milis.count() + 500) / 1000), colsize) << delim
                        << std::endl;
                }

                return ofs.good();
        }
}

