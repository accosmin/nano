#pragma once

#include <set>
#include <tuple>
#include <cmath>
#include <mutex>
#include <algorithm>

namespace ncv
{
        namespace tune_log10_detail
        {
                template
                <
                        typename tvalue,
                        typename tscalar,
                        typename tsize
                >
                void update_range(const tvalue& optimum, tscalar varlog, tsize splits, tscalar& minlog, tscalar& maxlog)
                {
                        minlog = std::get<1>(optimum) - varlog * tscalar(splits - 1) / tscalar(splits);
                        maxlog = std::get<1>(optimum) + varlog * tscalar(splits - 1) / tscalar(splits);
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
                        return std::make_tuple(std::get<0>(optimum), make_param(std::get<1>(optimum)));
                }

                template
                <
                        typename toperator,
                        typename tscalar
                >
                decltype(auto) evaluate(const toperator& op, tscalar log)
                {
                        return std::make_tuple(op(make_param(log)), log);
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
        decltype(auto) tune_log10(const toperator& op,
                tscalar minlog, tscalar maxlog, tscalar epslog, tsize splits)
        {
                typedef decltype(op(tscalar(0)))        tresult;
                typedef std::tuple<tresult, tscalar>    trecord;

                std::set<trecord> history;

                splits = std::max(tsize(4), splits);

                // greedy sort-of-branch-and-bound search
                while ((maxlog - minlog) > epslog && epslog > tscalar(0))
                {
                        const tscalar varlog = (maxlog - minlog) / tscalar(splits - 1);

                        for (tsize i = 0; i < splits; i ++)
                        {
                                const trecord value = tune_log10_detail::evaluate(op, minlog + i * varlog);

                                history.insert(value);
                        }
                        
                        tune_log10_detail::update_range(*history.begin(), varlog, splits, minlog, maxlog);
                }

                return history.empty() ? trecord() : tune_log10_detail::make_result(*history.begin());
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
        decltype(auto) tune_log10_mt(const toperator& op, tpool& pool,
                tscalar minlog, tscalar maxlog, tscalar epslog, tsize splits)
        {
                typedef decltype(op(tscalar(0)))        tresult;
                typedef std::tuple<tresult, tscalar>    trecord;

                std::set<trecord> history;

                // synchronization
                std::mutex mutex;
                
                splits = std::max(tsize(4), splits);
                
                // greedy sort-of-branch-and-bound search
                while ((maxlog - minlog) > epslog && epslog > tscalar(0))
                {
                        const tscalar varlog = (maxlog - minlog) / tscalar(splits - 1);

                        for (tsize i = 0; i < splits; i ++)
                        {
                                pool.enqueue([=,&history,&mutex]()
                                {
                                        const trecord value = tune_log10_detail::evaluate(op, minlog + i * varlog);

                                        // synchronize per thread
                                        const std::lock_guard<std::mutex> lock(mutex);
                                        history.insert(value);
                                });
                        }
                        
                        // synchronize per search step
                        pool.wait();
                        
                        tune_log10_detail::update_range(*history.begin(), varlog, splits, minlog, maxlog);
                }
                
                return history.empty() ? trecord() : tune_log10_detail::make_result(*history.begin());
        }
}
