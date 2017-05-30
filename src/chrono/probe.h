#pragma once

#include "measure.h"
#include "math/stats.h"

namespace nano
{
        ///
        /// \brief accumulate time measurements for a given operation of given complexity (aka flops).
        ///
        struct probe_t
        {
                using timings_t = stats_t<int64_t>;

                probe_t(const string_t& basename = string_t(),
                        const string_t& fullname = string_t(),
                        const int64_t flops = 1) :
                        m_basename(basename),
                        m_fullname(fullname),
                        m_flops(flops)
                {
                }

                template <typename toperator>
                void measure(const toperator& op)
                {
                        const timer_t timer;
                        op();
                        m_timings(timer.microseconds().count());
                }

                operator bool() const { return m_timings; }
                const auto& timings() const { return m_timings; }

                const auto& basename() const { return m_basename; }
                const auto& fullname() const { return m_fullname; }

                auto flops() const { return m_flops; }
                auto gflops() const { return nano::gflops(flops(), microseconds_t(timings().min())); }

                // attributes
                string_t        m_basename;             ///<
                string_t        m_fullname;             ///<
                int64_t         m_flops;                ///< number of floating point operations per call
                timings_t       m_timings;              ///< time measurements
        };

        using probes_t = std::vector<probe_t>;
}
