#pragma once

#include <cmath>
#include <limits>
#include <cassert>
#include <algorithm>
#include <type_traits>

namespace nano
{
        ///
        /// \brief computes statistics: average, standard deviation etc.
        ///
        template <typename tscalar = long double>
        class stats_t
        {
        public:

                static_assert(std::is_floating_point<tscalar>::value,
                        "stats_t can be used only with floating point types");

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
                        m_min = +std::numeric_limits<tscalar>::max();
                        m_max = -std::numeric_limits<tscalar>::max();
                }

                // access functions
                operator bool() const { return count() > 1; }

                std::size_t count() const { return m_count; }
                tscalar min() const { return m_min; }
                tscalar max() const { return m_max; }
                tscalar sum() const { return m_sum; }

                tscalar avg() const
                {
                        assert(count() > 0);
                        return sum() / static_cast<tscalar>(count());
                }

                tscalar var2() const
                {
                        assert(count() > 0);
                        return m_sumsq - m_sum * m_sum / static_cast<tscalar>(count());
                }

                tscalar var() const
                {
                        assert(count() > 0);
                        return (count() == 1) ? tscalar(0) : (var2() / static_cast<tscalar>(count() - 1));
                }

                tscalar stdev() const
                {
                        return std::sqrt(var());
                }

        private:

                // attributes
                std::size_t     m_count;
                tscalar         m_sum, m_sumsq;
                tscalar         m_min, m_max;
        };
}
