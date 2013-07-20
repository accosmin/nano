#ifndef NANOCV_STATS_H
#define NANOCV_STATS_H

#include "types.h"
#include <limits>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // computes statistics: average, standard deviation etc.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        class stats_t
	{
	public:
		
                // constructor
                stats_t()
                {
                        clear();
                }
		
                // reset statistics
                void clear()
                {
                        _clear();
                }
                
                // add new values
                void add(scalar_t value)
                {
                        _add(value);
                }

                void add(const stats_t& other)
                {
                        _add(other);
                }
                
                template
                <
                        class titerator
                >
                void add(titerator begin, titerator end)
		{
			for ( ; begin != end; ++ begin)
                        {
				add(*begin);
                        }
		}
		
                // access functions
                bool valid() const { return count() != 0; }
                size_t count() const { return m_count; }
                scalar_t min() const { return m_min; }
                scalar_t max() const { return m_max; }
                scalar_t avg() const { return _avg(); }
                scalar_t var() const { return _var(); }
                scalar_t stdev() const { return std::sqrt(_var()); }
                scalar_t sum() const { return m_sum1; }
                scalar_t sumsq() const { return m_sum2; }
		
	private:
                
                // reset statistics
                void _clear()
                { 
                        m_min = std::numeric_limits<scalar_t>::max();
                        m_max = -m_min;
                        m_sum1 = 0;
                        m_sum2 = 0;
                        m_count = 0;
                }
                
                // add new values
                void _add(scalar_t value)
                {
                        m_min = std::min(m_min, value);
                        m_max = std::max(m_max, value);
                        m_sum1 += value;
                        m_sum2 += value * value;
                        m_count ++;
                }

                void _add(const stats_t& other)
                {
                        m_min = std::min(m_min, other.m_min);
                        m_max = std::max(m_max, other.m_max);
                        m_sum1 += other.m_sum1;
                        m_sum2 += other.m_sum2;
                        m_count += other.m_count;
                }
                
                // access functions
                scalar_t _avg() const
                {
                        return count() < 1 ? sum() : sum() / count();
                }                
                scalar_t _var() const
                {
                        return count() < 2 ? 0.0 : (sumsq() - sum() * sum() / count()) / (count() - 1);
                }                
                
        private:
		 
                // attributes
                scalar_t        m_min, m_max;
                long double     m_sum1, m_sum2;
                size_t          m_count;
	};
}

#endif // NANOCV_STATS_H
