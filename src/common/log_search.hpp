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
        /// \returns the result associated with the optimum (log) paramerer.
        ///
        template
        <
                typename toperator,     ///< toperator(tscalar param) returns the optimal result for that parameter
                typename tscalar,
                typename tsize
        >
        auto log_min_search(const toperator& op, tscalar minlog, tscalar maxlog, tscalar epslog, tsize splits)
                -> decltype(op(tscalar(0)))
        {
                typedef decltype(op(tscalar(0)))        tresult;                
                typedef std::pair<tresult, tscalar>     tvalue;
                
                std::set<tvalue> history;

                splits = std::max(tsize(4), splits);
                
                for (   tscalar varlog = (maxlog - minlog) / tscalar(splits - 1);
                        (maxlog - minlog) > epslog && epslog > tscalar(0); 
                        varlog = varlog / splits)
                {
                        for (tsize i = 0; i < splits; i ++)
                        {
                                const tscalar log = minlog + i * varlog;
                                history.insert(std::make_pair(op(std::exp(log)), log));
                        }
                        
                        const tvalue& optimum = *history.begin();
                        minlog = optimum.second - varlog * tscalar(splits - 1) / tscalar(splits);
                        maxlog = optimum.second + varlog * tscalar(splits - 1) / tscalar(splits);
                }
                
                return history.empty() ? tresult() : history.begin()->first;
        }
        
        ///
        /// \brief multi-threaded search for a 1D parameter that minimizes a given operator, 
        ///     using a greedy approach on the logarithmic scale in the range [minlog, maxlog].
        ///
        /// \returns the result associated with the optimum (log) paramerer.
        ///
        template
        <
                typename toperator,     ///< toperator(tscalar param) returns the optimal result for that parameter
                typename tpool,         ///< thread pool
                typename tscalar,
                typename tsize
        >
        auto log_min_search_mt(const toperator& op, tpool& pool, tscalar minlog, tscalar maxlog, tscalar epslog, tsize splits)
                -> decltype(op(tscalar(0)))
        {
                typedef decltype(op(tscalar(0)))        tresult;                
                typedef std::pair<tresult, tscalar>     tvalue;
                
                std::set<tvalue> history;
                std::vector<tvalue> results(splits);
                
                splits = std::max(tsize(4), splits);
                
                for (   tscalar varlog = (maxlog - minlog) / tscalar(splits - 1);
                        (maxlog - minlog) > epslog && epslog > tscalar(0); 
                        varlog = varlog / splits)
                {                                                
                        for (tsize i = 0; i < splits; i ++)
                        {
                                const tscalar log = minlog + i * varlog;
                                pool.enqueue([=,&results,&op]()
                                {
                                        results[i] = std::make_pair(op(std::exp(log)), log);
                                });
                        }
                        
                        pool.wait();    // synchronize
                        
                        history.insert(results.begin(), results.end()); // update history
                        
                        const tvalue& optimum = *history.begin();
                        minlog = optimum.second - varlog * tscalar(splits - 1) / tscalar(splits);
                        maxlog = optimum.second + varlog * tscalar(splits - 1) / tscalar(splits);
                }
                
                return history.empty() ? tresult() : history.begin()->first;
        }
}
