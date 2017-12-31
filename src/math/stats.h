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
                /// \brief update statistics with new values
                ///
                void operator()(const tscalar value)
                {
                        if (std::isfinite(value))
                        {
                                m_avg1 += (value - m_avg1) / (m_count + 1);
                                m_avg2 += (value * value - m_avg2) / (m_count + 1);
                                m_min = std::min(m_min, value);
                                m_max = std::max(m_max, value);
                                m_count ++;
                        }
                }

                template <typename... tscalars>
                void operator()(const tscalar value, const tscalars... values)
                {
                        operator()(value);
                        operator()(values...);
                }

                ///
                /// \brief merge statistics
                ///
                void operator()(const stats_t& other)
                {
                        m_avg1 = (m_avg1 * m_count + other.m_avg1 * other.m_count) / (m_count + other.m_count);
                        m_avg2 = (m_avg2 * m_count + other.m_avg2 * other.m_count) / (m_count + other.m_count);
                        m_count += other.m_count;
                        m_min = std::min(m_min, other.m_min);
                        m_max = std::max(m_max, other.m_max);
                }

                ///
                /// \brief update statistics with the given [begin, end) range
                ///
                template <typename titerator, typename = typename std::iterator_traits<titerator>::value_type>
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
                        m_avg1 = 0;
                        m_avg2 = 0;
                        m_min = std::numeric_limits<tscalar>::max();
                        m_max = std::numeric_limits<tscalar>::lowest();
                }

                ///
                /// \brief returns average
                ///
                tscalar avg() const
                {
                        assert(count() > 0);
                        return static_cast<tscalar>(m_avg1);
                }

                ///
                /// \brief return variance
                ///
                tscalar var() const
                {
                        assert(count() > 0);
                        return static_cast<tscalar>(m_avg2 - m_avg1 * m_avg1);
                }

                ///
                /// \brief returns population standard deviation
                ///
                tscalar stdev() const
                {
                        return std::sqrt(var());
                }

                // access functions
                operator bool() const { return count() > 1; }

                std::size_t count() const { return m_count; }
                tscalar min() const { return m_min; }
                tscalar max() const { return m_max; }
                tscalar sum1() const { return static_cast<tscalar>(m_avg1 * m_count); }
                tscalar sum2() const { return static_cast<tscalar>(m_avg2 * m_count); }

        private:

                // attributes
                std::size_t     m_count{0};
                tstorage        m_avg1{0}, m_avg2{0};
                tscalar         m_min, m_max;
        };

        template <typename tscalar>
        std::ostream& operator<<(std::ostream& os, const stats_t<tscalar>& stats)
        {
                return  !stats ?
                        os :
                        (os << stats.avg() << "+/-" << stats.stdev() << "[" << stats.min() << "," << stats.max() << "]");
        }
}
