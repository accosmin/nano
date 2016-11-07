#include "math/stats.h"
#include "text/align.h"
#include "math/epsilon.h"
#include "trainer_state.h"
#include "text/to_string.h"
#include <fstream>

namespace nano
{
        trainer_measurement_t::trainer_measurement_t() : trainer_measurement_t(
                std::numeric_limits<scalar_t>::max(),
                std::numeric_limits<scalar_t>::max(),
                std::numeric_limits<scalar_t>::max(),
                std::numeric_limits<scalar_t>::max(),
                std::numeric_limits<scalar_t>::max(),
                std::numeric_limits<scalar_t>::max(),
                std::numeric_limits<scalar_t>::max())
        {
        }

        trainer_measurement_t::trainer_measurement_t(
                const scalar_t value,
                const scalar_t value_avg, const scalar_t value_var, const scalar_t value_max,
                const scalar_t error_avg, const scalar_t error_var, const scalar_t error_max) :
                m_value(value),
                m_value_avg(value_avg), m_value_var(value_var), m_value_max(value_max),
                m_error_avg(error_avg), m_error_var(error_var), m_error_max(error_max)
        {
        }

        trainer_measurement_t::trainer_measurement_t(
                const scalar_t value,
                const stats_t<scalar_t>& vstats, const stats_t<scalar_t>& estats) : trainer_measurement_t(
                value,
                vstats.avg(), vstats.var(), vstats.max(),
                estats.avg(), estats.var(), estats.max())
        {
        }

        trainer_measurement_t::operator bool() const
        {
                return  std::isfinite(m_value) &&
                        std::isfinite(m_value_avg) && std::isfinite(m_value_var) &&
                        std::isfinite(m_error_avg) && std::isfinite(m_error_var);
        }

        bool operator<(const trainer_measurement_t& one, const trainer_measurement_t& two)
        {
                const auto v1 = (one) ? one.m_value_avg : std::numeric_limits<scalar_t>::max();
                const auto v2 = (two) ? two.m_value_avg : std::numeric_limits<scalar_t>::max();
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

                const string_t delim = "    ";
                const size_t colsize = 24;

                // header
                for (const string_t& proto : {"train", "valid", "test"})
                {
                        ofs
                        << nano::align(proto + "-criterion", colsize) << delim
                        << nano::align(proto + "-loss-avg", colsize) << delim
                        << nano::align(proto + "-loss-var", colsize) << delim
                        << nano::align(proto + "-loss-max", colsize) << delim
                        << nano::align(proto + "-error-avg", colsize) << delim
                        << nano::align(proto + "-error-var", colsize) << delim
                        << nano::align(proto + "-error-max", colsize) << delim;
                }
                ofs
                << nano::align("time-seconds", colsize) << delim
                << std::endl;

                // optimization states
                for (const trainer_state_t& state : states)
                {
                        for (const auto& measurement : {state.m_train, state.m_valid, state.m_test})
                        {
                                ofs
                                << nano::align(nano::to_string(measurement.m_value), colsize) << delim
                                << nano::align(nano::to_string(measurement.m_value_avg), colsize) << delim
                                << nano::align(nano::to_string(measurement.m_value_var), colsize) << delim
                                << nano::align(nano::to_string(measurement.m_value_max), colsize) << delim
                                << nano::align(nano::to_string(measurement.m_error_avg), colsize) << delim
                                << nano::align(nano::to_string(measurement.m_error_var), colsize) << delim
                                << nano::align(nano::to_string(measurement.m_error_max), colsize) << delim;
                        }
                        ofs
                        << nano::align(nano::to_string((state.m_milis.count() + 500) / 1000), colsize) << delim
                        << std::endl;
                }

                return ofs.good();
        }
}

