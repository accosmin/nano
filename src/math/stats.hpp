#pragma once

#include <cmath>
#include <limits>
#include <cassert>
#include <algorithm>

namespace math
{
        ///
        /// \brief computes statistics: average, standard deviation etc.
        ///
        template
        <
                typename tscalar = double,
                typename tsize = std::size_t
        >
        class stats_t
        {
        public:

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
                void operator()(tscalar value)
                {
                        m_count ++;
                        m_sum += value;
                        m_sumsq += value * value;
                        m_min = std::min(m_min, value);
                        m_max = std::max(m_max, value);
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
                template
                <
                        class titerator
                >
                void operator()(titerator begin, titerator end)
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
                        m_min = +std::numeric_limits<tscalar>::max();
                        m_max = -std::numeric_limits<tscalar>::max();
                }

                // access functions
                operator bool() const { return count() > 1; }

                tsize count() const { return m_count; }
                tscalar min() const { return m_min; }
                tscalar max() const { return m_max; }
                tscalar sum() const { return m_sum; }

                double avg() const
                {
                        assert(count() > 0);
                        return  static_cast<double>(sum()) /
                                static_cast<double>(count());
                }

                double var2() const
                {
                        assert(count() > 0);
                        return  static_cast<double>(m_sumsq) -
                                static_cast<double>(m_sum * m_sum) / static_cast<double>(count());
                }

                double var() const
                {
                        assert(count() > 0);
                        return (count() == 1) ? 0.0 : (var2() / static_cast<double>(count() - 1));
                }

                double stdev() const
                {
                        return std::sqrt(var());
                }

        private:

                // attributes
                tsize           m_count;
                tscalar         m_sum, m_sumsq;
                tscalar         m_min, m_max;
        };
}
