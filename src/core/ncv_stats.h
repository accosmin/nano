#ifndef NANOCV_STATS_H
#define NANOCV_STATS_H

#include "ncv_types.h"
#include "ncv_math.h"
#include <limits>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // computes statistics: average, standard deviation etc.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
	
        template
        <
                typename tscalar = double
        >
        class stats
	{
	public:
		
                // constructor
                stats() { _clear(); }
		
                // reset statistics
		void clear() { _clear(); }
                
                // add new values
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
		
                // access functions
		bool            valid() const { return count() != 0; }
                size_t          count() const { return m_count; }
                tscalar         min() const { return m_min; }
                tscalar         max() const { return m_max; }
                tscalar         avg() const { return _avg(); }
                tscalar         stdev() const { return _stdev(); }
                tscalar         sum() const { return m_sum1; }
                tscalar         sumsq() const { return m_sum2; }
		
	private:
                
                // reset statistics
                void _clear()
                { 
                        m_min = std::numeric_limits<tscalar>::max();
                        m_max = -m_min;
                        m_sum1 = 0;
                        m_sum2 = 0;
                        m_count = 0;
                }
                
                // add new values
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
                
                // access functions
                tscalar _avg() const
                {
                        return  count() < 1 ? sum() : sum() / count();
                }                
                tscalar _stdev() const
                {
                        return  count() < 2 ? 
                                0.0l : std::sqrt((sumsq() - sum() * sum() / count()) / (count() - 1));
                }                
                
        private:
		 
                // attributes
                tscalar         m_min, m_max;
                tscalar         m_sum1, m_sum2;
                size_t          m_count;
	};
}

#endif // NANOCV_STATS_H
