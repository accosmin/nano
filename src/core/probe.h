#pragma once

#include <vector>
#include "stats.h"
#include "measure.h"

namespace nano
{
        ///
        /// \brief accumulate time measurements for a given operation of given complexity (aka flops).
        ///
        class probe_t
        {
        public:

                using timings_t = stats_t<int64_t>;

                probe_t(const std::string& basename = std::string(),
                        const std::string& fullname = std::string(),
                        const int64_t flops = 1) :
                        m_basename(basename),
                        m_fullname(fullname),
                        m_flops(flops)
                {
                }

                template <typename toperator>
                void measure(const toperator& op, const int64_t count = 1)
                {
                        assert(count > 0);
                        const timer_t timer;
                        op();
                        m_timings(timer.nanoseconds().count() / count);
                }

                operator bool() const { return m_timings; }
                const auto& timings() const { return m_timings; }

                const auto& basename() const { return m_basename; }
                const auto& fullname() const { return m_fullname; }

                auto flops() const { return m_flops; }
                auto kflops() const { return m_flops / 1024; }
                auto gflops() const { return nano::gflops(flops(), nanoseconds_t(timings().min())); }

        private:

                // attributes
                std::string     m_basename;             ///<
                std::string     m_fullname;             ///<
                int64_t         m_flops;                ///< number of floating point operations per call
                timings_t       m_timings;              ///< time measurements
        };

        using probes_t = std::vector<probe_t>;
}
