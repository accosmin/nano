#pragma once

#include <limits>
#include <cmath>
#include <type_traits>
#include <algorithm>

namespace ncv
{
        ///
        /// computes statistics: average, standard deviation etc.
        ///
        template
        <
                typename tscalar = double,
                typename tsize = std::size_t,

                // disable for not valid types!
                typename tvalid_tscalar = typename std::enable_if<std::is_floating_point<tscalar>::value>::type,
                typename tvalid_tsize = typename std::enable_if<std::is_integral<tsize>::value>::type
        >
        class stats_t
	{
        public:

                // constructor
                stats_t()
                        :       m_count(0),
                                m_sum(0),
                                m_sumsq(0),
                                m_min(+std::numeric_limits<tscalar>::max()),
                                m_max(-std::numeric_limits<tscalar>::max())
                {
                }

                // add new values
                void operator()(tscalar value)
                {
                        m_count ++;
                        m_sum += value;
                        m_sumsq += value * value;
                        m_min = std::min(m_min, value);
                        m_max = std::max(m_max, value);
                }

                void operator()(const stats_t& other)
                {
                        m_count += other.m_count;
                        m_sum += other.m_sum;
                        m_sumsq += other.m_sumsq;
                        m_min = std::min(m_min, other.m_min);
                        m_max = std::max(m_max, other.m_max);
                }
                
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
		
                // access functions
                bool valid() const { return count() != 0; }
                tsize count() const { return m_count; }
                tscalar min() const { return m_min; }
                tscalar max() const { return m_max; }
                tscalar avg() const { return sum() / count(); }
                tscalar var() const { return _var() / m_count; }
                tscalar stdev() const { return std::sqrt(m_count > 1 ? _var() / (m_count - 1) : tscalar(0)); }
                tscalar sum() const { return m_sum; }

        private:

                tscalar _var() const
                {
                        return (m_count == 0) ? tscalar(0) : (m_sumsq - m_sum * m_sum / m_count);
                }
                
        private:

                // attributes
                tsize           m_count;
                tscalar         m_sum, m_sumsq;
                tscalar         m_min, m_max;
	};
}
