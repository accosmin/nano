#pragma once

#include <mutex>
#include "tune_log10.hpp"

namespace math
{
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
        auto tune_log10_mt(const toperator& op, tpool& pool,
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
