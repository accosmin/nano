#ifndef NANOCV_STATS_H
#define NANOCV_STATS_H

#include "ncv_types.h"
#include "ncv_math.h"
#include <limits>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Computes statistics: average, standard deviation etc.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
	
        template
        <
                typename tscalar = double
        >
        class stats
	{
	public:
		
		// Constructor
                stats() { _clear(); }
		
		// Reset statistics
		void clear() { _clear(); }
                
                // Add new values
                void add(tscalar value) { _add(value); }

                template <typename tscalar_other>
                void add(const stats<tscalar_other>& other) { _add(other); }
                
                template <class TIterator>
                void add(TIterator begin, TIterator end)
		{
			for ( ; begin != end; ++ begin)
                        {
				add(*begin);
                        }
		}
		
		// Access functions
		bool            valid() const { return count() != 0; }
                size_t          count() const { return m_count; }
                tscalar         min() const { return m_min; }
                tscalar         max() const { return m_max; }
                tscalar         avg() const { return _avg(); }
                tscalar         stdev() const { return _stdev(); }
                tscalar         sum() const { return m_sum1; }
                tscalar         sumsq() const { return m_sum2; }
		
	private:
                
                // Reset statistics
                void _clear()
                { 
                        m_min = std::numeric_limits<tscalar>::max();
                        m_max = -m_min;
                        m_sum1 = 0;
                        m_sum2 = 0;
                        m_count = 0;
                }
                
                // Add new values
                void _add(tscalar value)
                {
                        m_min = std::min(m_min, value);
                        m_max = std::max(m_max, value);
                        m_sum1 += value;
                        m_sum2 += value * value;
                        m_count ++;
                }

                template <typename tscalar_other>
                void _add(const stats<tscalar_other>& other)
                {
                        m_min = std::min(m_min, static_cast<tscalar>(other.m_min));
                        m_max = std::max(m_max, static_cast<tscalar>(other.m_max));
                        m_sum1 += static_cast<tscalar>(other.m_sum1);
                        m_sum2 += static_cast<tscalar>(other.m_sum2);
                        m_count += other.m_count;
                }
                
                // Access functions
                tscalar _avg() const
                {
                        return math::inverse(count()) * sum();
                }                
                tscalar _stdev() const
                {
                        return  count() < 2 ? 
                                0 :
                                std::sqrt(math::inverse(count() - 1) *
                                (sumsq() - math::inverse(count()) * sum() * sum()));
                }                
                
        private:
		 
		// Attributes
                tscalar         m_min, m_max;
                tscalar         m_sum1, m_sum2;
                size_t          m_count;
	};
}

#endif // NANOCV_STATS_H
