#pragma once

#include <cmath>
#include <set>
#include <algorithm>
#include <vector>

namespace ncv
{
        namespace min_search_detail
        {
                template
                <
                        typename tvalue,
                        typename tsize,
                        typename tscalar
                >
                void update_range(const tvalue& optimum, tsize splits, tscalar varlog, tscalar& minlog, tscalar& maxlog)
                {
                        minlog = optimum.second - varlog * tscalar(splits - 1) / tscalar(splits);
                        maxlog = optimum.second + varlog * tscalar(splits - 1) / tscalar(splits);
                }

                template
                <
                        typename tscalar
                >
                tscalar make_param(tscalar log)
                {
                        return std::pow(tscalar(10), log);
                }

                template
                <
                        typename tvalue
                >
                tvalue make_result(const tvalue& optimum)
                {
                        return std::make_pair(optimum.first, make_param(optimum.second));
                }

                template
                <
                        typename toperator,
                        typename tscalar
                >
                auto evaluate(const toperator& op, tscalar log) -> std::pair<decltype(op(tscalar(0))), tscalar>
                {
                        return std::make_pair(op(make_param(log)), log);
                }
        }

        ///
        /// \brief search for a 1D parameter that minimizes a given operator, 
        ///     using a greedy approach on the base 10 logarithmic scale in the range [minlog, maxlog].
        ///
        /// \returns { the result associated to the optimum parameter, the optimum parameter }.
        ///
        template
        <
                typename toperator,     ///< toperator(tscalar param) returns the optimal result for that parameter
                typename tscalar,
                typename tsize
        >
        auto log10_min_search(const toperator& op,
                tscalar minlog, tscalar maxlog, tscalar epslog, tsize splits)
                -> std::pair<decltype(op(tscalar(0))), tscalar>
        {
                // <result, logarithmic parameter>
                typedef decltype(op(tscalar(0)))        tresult;
                typedef std::pair<tresult, tscalar>     tvalue;
                
                std::set<tvalue> history;

                splits = std::max(tsize(4), splits);

                // greedy sort-of-branch-and-bound search
                for (   tscalar varlog = (maxlog - minlog) / tscalar(splits - 1);
                        (maxlog - minlog) > epslog && epslog > tscalar(0);
                        varlog = varlog / tscalar(splits))
                {
                        for (tsize i = 0; i < splits; i ++)
                        {
                                const tvalue value = min_search_detail::evaluate(op, minlog + i * varlog);

                                history.insert(value);
                        }
                        
                        min_search_detail::update_range(*history.begin(), splits, varlog, minlog, maxlog);
                }

                return history.empty() ? tvalue() : min_search_detail::make_result(*history.begin());
        }
        
        ///
        /// \brief multi-threaded search for a 1D parameter that minimizes a given operator, 
        ///     using a greedy approach on the base 10 logarithmic scale in the range [minlog, maxlog].
        ///
        /// \returns { the result associated to the optimum parameter, the optimum parameter }.
        ///
        template
        <
                typename toperator,     ///< toperator(tscalar param) returns the optimal result for that parameter
                typename tpool,         ///< thread pool
                typename tscalar,
                typename tsize
        >
        auto log10_min_search_mt(const toperator& op, tpool& pool,
                tscalar minlog, tscalar maxlog, tscalar epslog, tsize splits)
                -> std::pair<decltype(op(tscalar(0))), tscalar>
        {
                // <result, logarithmic parameter>
                typedef decltype(op(tscalar(0)))        tresult;                
                typedef std::pair<tresult, tscalar>     tvalue;

                std::set<tvalue> history;

                // synchronization
                typedef typename tpool::mutex_t         tmutex;
                typedef typename tpool::lock_t          tlock;

                tmutex mutex;
                
                splits = std::max(tsize(4), splits);
                
                // greedy sort-of-branch-and-bound search
                for (   tscalar varlog = (maxlog - minlog) / tscalar(splits - 1);
                        (maxlog - minlog) > epslog && epslog > tscalar(0); 
                        varlog = varlog / tscalar(splits))
                {                                                
                        for (tsize i = 0; i < splits; i ++)
                        {
                                pool.enqueue([=,&history,&mutex]()
                                {
                                        const tvalue value = min_search_detail::evaluate(op, minlog + i * varlog);

                                        // synchronize per thread
                                        const tlock lock(mutex);
                                        history.insert(value);
                                });
                        }
                        
                        // synchronize per search step
                        pool.wait();
                        
                        min_search_detail::update_range(*history.begin(), splits, varlog, minlog, maxlog);
                }
                
                return history.empty() ? tvalue() : min_search_detail::make_result(*history.begin());
        }
}
