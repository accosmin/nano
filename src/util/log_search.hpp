#pragma once

#include <cmath>
#include <set>
#include <algorithm>
#include <vector>

namespace ncv
{
        ///
        /// \brief search for a 1D parameter that minimizes a given operator, 
        ///     using a greedy approach on the logarithmic scale in the range [minlog, maxlog].
        ///
        /// \returns { the result associated to the optimum (log) parameter, the optimum (log) parameter }.
        ///
        template
        <
                typename toperator,     ///< toperator(tscalar param) returns the optimal result for that parameter
                typename tscalar,
                typename tsize
        >
        auto log_min_search(const toperator& op, tscalar minlog, tscalar maxlog, tscalar epslog, tsize splits)
                -> std::pair<decltype(op(tscalar(0))), tscalar>
        {
                typedef decltype(op(tscalar(0)))        tresult;                
                typedef std::pair<tresult, tscalar>     tvalue;
                
                std::set<tvalue> history;

                splits = std::max(tsize(4), splits);
                
                // greedy sort-of-branch-and-bound search
                for (   tscalar varlog = (maxlog - minlog) / tscalar(splits - 1);
                        (maxlog - minlog) > epslog && epslog > tscalar(0); 
                        varlog = varlog / splits)
                {
                        for (tsize i = 0; i < splits; i ++)
                        {
                                const tscalar log = minlog + i * varlog;
                                const tresult result = op(std::exp(log));

                                history.insert(std::make_pair(result, log));
                        }
                        
                        const tvalue& optimum = *history.begin();
                        minlog = optimum.second - varlog * tscalar(splits - 1) / tscalar(splits);
                        maxlog = optimum.second + varlog * tscalar(splits - 1) / tscalar(splits);
                }

                return history.empty() ? tvalue() : *history.begin();
        }
        
        ///
        /// \brief multi-threaded search for a 1D parameter that minimizes a given operator, 
        ///     using a greedy approach on the logarithmic scale in the range [minlog, maxlog].
        ///
        /// \returns { the result associated to the optimum (log) parameter, the optimum (log) parameter }.
        ///
        template
        <
                typename toperator,     ///< toperator(tscalar param) returns the optimal result for that parameter
                typename tpool,         ///< thread pool
                typename tscalar,
                typename tsize
        >
        auto log_min_search_mt(const toperator& op, tpool& pool, tscalar minlog, tscalar maxlog, tscalar epslog, tsize splits)
                -> std::pair<decltype(op(tscalar(0))), tscalar>
        {
                typedef decltype(op(tscalar(0)))        tresult;                
                typedef std::pair<tresult, tscalar>     tvalue;
                typedef typename tpool::mutex_t         tmutex;
                typedef typename tpool::lock_t          tlock;
                
                std::set<tvalue> history;
                tmutex mutex;
                
                splits = std::max(tsize(4), splits);
                
                // greedy sort-of-branch-and-bound search
                for (   tscalar varlog = (maxlog - minlog) / tscalar(splits - 1);
                        (maxlog - minlog) > epslog && epslog > tscalar(0); 
                        varlog = varlog / splits)
                {                                                
                        for (tsize i = 0; i < splits; i ++)
                        {
                                pool.enqueue([=,&history,&mutex,&op]()
                                {
                                        const tscalar log = minlog + i * varlog;
                                        const tresult result = op(std::exp(log));

                                        // synchronize per thread
                                        const tlock lock(mutex);
                                        history.insert(std::make_pair(result, log));
                                });
                        }
                        
                        // synchronize per search step
                        pool.wait();
                        
                        const tvalue& optimum = *history.begin();
                        minlog = optimum.second - varlog * tscalar(splits - 1) / tscalar(splits);
                        maxlog = optimum.second + varlog * tscalar(splits - 1) / tscalar(splits);
                }
                
                return history.empty() ? tvalue() : *history.begin();
        }
}
