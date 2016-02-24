#pragma once

#include <set>
#include <cassert>
#include <algorithm>
#include "tune_log10_detail.hpp"

namespace math
{
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
        auto tune_log10(const toperator& op,
                tscalar minlog, tscalar maxlog, tscalar epslog, tsize splits)
        {
                using tresult = decltype(op(tscalar(0)));
                using trecord = std::tuple<tresult, tscalar>;

                splits = std::max(tsize(4), splits);

                // greedy sort-of-branch-and-bound search
                std::set<trecord> history;
                while ((maxlog - minlog) > epslog && epslog > tscalar(0))
                {
                        const auto varlog = (maxlog - minlog) / tscalar(splits - 1);

                        for (tsize i = 0; i < splits; i ++)
                        {
                                const trecord value = tune_log10_detail::evaluate(op, minlog, i, varlog);
                                history.insert(value);
                        }

                        tune_log10_detail::update_range(*history.begin(), varlog, splits, minlog, maxlog);
                }

                assert(!history.empty());
                return tune_log10_detail::make_result(*history.begin());
        }
}
