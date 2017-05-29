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

                probe_t(const string_t& name = string_t(), const int64_t flops = 1) :
                        m_name(name),
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

                const auto& name() const { return m_name; }
                const auto gflops() const { return nano::gflops(m_flops, timings.min()); }
                const auto& timings() const { return m_timings; }

                // attributes
                const string_t  m_name;                 ///< name
                const int64_t   m_flops;                ///< #floating point operations per call
                timings_t       m_timings;              ///< time measurements
        };

        using probes_t = std::vector<probe_t>;
}
