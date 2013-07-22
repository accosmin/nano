#ifndef NANOCV_STATS_H
#define NANOCV_STATS_H

#include <limits>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // computes statistics: average, standard deviation etc.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar = double
        >
        class stats_t
	{
        public:
		
                // reset statistics
                void clear() { _clear(); }
                
                // add new values
                void add(tscalar value)
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
                size_t count() const { return boost::accumulators::count(m_acc); }
                tscalar min() const { return boost::accumulators::min(m_acc); }
                tscalar max() const { return boost::accumulators::max(m_acc); }
                tscalar avg() const { return boost::accumulators::mean(m_acc); }
                tscalar var() const { return boost::accumulators::variance(m_acc)(); }
                tscalar stdev() const { return std::sqrt(var()); }
		
        private:

                // reset statistics
                void _clear()
                { 
                        m_acc.drop<boost::accumulators::tag::min>();
                        m_acc.drop<boost::accumulators::tag::max>();
                        m_acc.drop<boost::accumulators::tag::mean>();
                        m_acc.drop<boost::accumulators::tag::variance>();
                }
                
                // add new values
                void _add(tscalar value)
                {
                        m_acc(value);
                }

                void _add(const stats_t& other)
                {
                        m_acc(other.m_acc);
                }
                
        private:

                typedef boost::accumulators::accumulator_set
                <
                        tscalar,
                        boost::accumulators::stats
                        <
                                boost::accumulators::droppable<boost::accumulators::tag::min>,
                                boost::accumulators::droppable<boost::accumulators::tag::max>,
                                boost::accumulators::droppable<boost::accumulators::tag::mean>,
                                boost::accumulators::droppable<boost::accumulators::tag::variance>
                        >
                >       accumulator_t;
		 
                // attributes
                accumulator_t   m_acc;
	};
}

#endif // NANOCV_STATS_H
