#pragma once

#include <cmath>
#include <limits>
#include <cassert>
#include <ostream>
#include <algorithm>

namespace nano
{
        ///
        /// \brief computes statistics: average, standard deviation etc.
        ///
        template <typename tscalar = long double>
        class stats_t
        {
        public:

                using tstorage = long double;

                ///
                /// \brief constructor
                ///
                stats_t()
                {
                        clear();
                }

                ///
                /// \brief update statistics with a new value
                ///
                void operator()(const tscalar value)
                {
                        if (std::isfinite(value))
                        {
                                m_count ++;
                                m_sum += value;
                                m_sumsq += value * value;
                                m_min = std::min(m_min, value);
                                m_max = std::max(m_max, value);
                        }
                }

                ///
                /// \brief merge statistics
                ///
                void operator()(const stats_t& other)
                {
                        m_count += other.m_count;
                        m_sum += other.m_sum;
                        m_sumsq += other.m_sumsq;
                        m_min = std::min(m_min, other.m_min);
                        m_max = std::max(m_max, other.m_max);
                }

                ///
                /// \brief update statistics with the given [begin, end) range
                ///
                template <typename titerator>
                void operator()(titerator begin, const titerator end)
                {
                        for ( ; begin != end; ++ begin)
                        {
                                operator()(*begin);
                        }
                }

                ///
                /// \brief reset statistics
                ///
                void clear()
                {
                        m_count = 0;
                        m_sum = 0;
                        m_sumsq = 0;
                        m_min = std::numeric_limits<tscalar>::max();
                        m_max = std::numeric_limits<tscalar>::lowest();
                }

                // access functions
                operator bool() const { return count() > 1; }

                std::size_t count() const { return m_count; }
                tstorage count1() const { return static_cast<tstorage>(count() - 1); }
                tscalar min() const { return m_min; }
                tscalar max() const { return m_max; }
                tstorage sum() const { return m_sum; }

                tscalar avg() const
                {
                        assert(count() > 0);
                        return static_cast<tscalar>(sum() / count());
                }

                tscalar var2() const
                {
                        assert(count() > 0);
                        return static_cast<tscalar>(m_sumsq - m_sum * m_sum / count());
                }

                tscalar var() const
                {
                        assert(count() > 0);
                        return (count() == 1) ? tscalar(0) : static_cast<tscalar>(var2() / count1());
                }

                tscalar stdev() const
                {
                        assert(count() > 0);
                        return (count() == 1) ? tscalar(0) : static_cast<tscalar>(std::sqrt(var2() / count1()));
                }

        private:

                // attributes
                std::size_t     m_count;
                tstorage        m_sum, m_sumsq;
                tscalar         m_min, m_max;
        };

        template <typename tscalar>
        std::ostream& operator<<(std::ostream& os, const stats_t<tscalar>& stats)
        {
                return os << stats.avg() << " +/- " << stats.stdev() << " [" << stats.min() << ", " << stats.max() << "]";
        }
}
