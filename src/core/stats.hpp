#ifndef NANOCV_STATS_H
#define NANOCV_STATS_H

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // computes statistics: average, standard deviation etc.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar = double,
                typename tsize = std::size_t
        >
        class stats_t
	{
        public:

                // add new values
                void add(tscalar value)
                {
                        m_acc(value);
                }

                void add(const stats_t& other)
                {
                        // FIXME: how to combine these two accumulators?!
                        // => go back to own implementation!
//                        m_acc(other.m_acc);
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
                tsize count() const { return static_cast<tsize>(boost::accumulators::count(m_acc)); }
                tscalar min() const { return boost::accumulators::min(m_acc); }
                tscalar max() const { return boost::accumulators::max(m_acc); }
                tscalar avg() const { return boost::accumulators::mean(m_acc); }
                tscalar var() const { return boost::accumulators::variance(m_acc); }
                tscalar stdev() const { return std::sqrt(var()); }
                tscalar sum() const { return boost::accumulators::sum(m_acc); }
                
        private:

                typedef boost::accumulators::accumulator_set
                <
                        tscalar,
                        boost::accumulators::stats
                        <
                                boost::accumulators::tag::min,
                                boost::accumulators::tag::max,
                                boost::accumulators::tag::variance
                        >
                >       accumulator_t;
		 
                // attributes
                accumulator_t   m_acc;
	};
}

#endif // NANOCV_STATS_H
