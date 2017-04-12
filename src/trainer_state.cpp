#include "math/stats.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "trainer_state.h"
#include "text/algorithm.h"
#include "text/to_string.h"
#include <fstream>

namespace nano
{
        trainer_measurement_t::trainer_measurement_t() : trainer_measurement_t(
                std::numeric_limits<scalar_t>::max(),
                std::numeric_limits<scalar_t>::max())
        {
        }

        trainer_measurement_t::trainer_measurement_t(
                const scalar_t value, const scalar_t error) :
                m_value(value), m_error(error)
        {
        }

        trainer_measurement_t::operator bool() const
        {
                return  std::isfinite(m_value) &&
                        std::isfinite(m_value);
        }

        bool operator<(const trainer_measurement_t& one, const trainer_measurement_t& two)
        {
                const auto v1 = (one) ? one.m_value : std::numeric_limits<scalar_t>::max();
                const auto v2 = (two) ? two.m_value : std::numeric_limits<scalar_t>::max();
                return v1 < v2;
        }

        trainer_state_t::trainer_state_t() :
                m_milis(0),
                m_epoch(0),
                m_xnorm(0),
                m_gnorm(0)
        {
        }

        trainer_state_t::trainer_state_t(
                const milliseconds_t milis,
                const size_t epoch,
                const scalar_t xnorm,
                const scalar_t gnorm,
                const trainer_measurement_t& train,
                const trainer_measurement_t& valid,
                const trainer_measurement_t& test) :
                m_milis(milis),
                m_epoch(epoch),
                m_xnorm(xnorm),
                m_gnorm(gnorm),
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

                        const auto ratio_eps = nano::epsilon0<scalar_t>();
                        const auto ratio = (ratio_eps + crt.m_train.m_value) / (ratio_eps + prv.m_train.m_value);
                        const auto delta = 1 + crt.m_milis.count() - prv.m_milis.count();

                        // convergence speed ~ loss decrease ratio / second
                        const auto ret = static_cast<scalar_t>(std::pow(
                                static_cast<double>(ratio),
                                static_cast<double>(1000) / static_cast<double>(delta)));
                        return std::isfinite(ret) ? nano::clamp(ret, scalar_t(0), scalar_t(1)) : scalar_t(1);
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
                ofs << nano::align("epoch", colsize) << delim;
                for (const string_t& proto : {"train", "valid", "test"})
                {
                        ofs
                        << nano::align(proto + "_loss", colsize) << delim
                        << nano::align(proto + "_error", colsize) << delim;
                }
                ofs
                << nano::align("seconds", colsize) << delim
                << nano::align("xnorm", colsize) << delim
                << nano::align("gnorm", colsize) << delim
                << std::endl;

                // optimization states
                size_t index = 0;
                for (const trainer_state_t& state : states)
                {
                        ofs << nano::align(to_string(index ++), colsize) << delim;
                        for (const auto& measurement : {state.m_train, state.m_valid, state.m_test})
                        {
                                ofs
                                << nano::align(to_string(measurement.m_value), colsize) << delim
                                << nano::align(to_string(measurement.m_error), colsize) << delim;
                        }
                        ofs
                        << nano::align(to_string(idiv(state.m_milis.count(), 1000)), colsize) << delim
                        << nano::align(to_string(state.m_xnorm), colsize) << delim
                        << nano::align(to_string(state.m_gnorm), colsize) << delim
                        << std::endl;
                }

                return ofs.good();
        }
}

