#pragma once

#include <tuple>
#include <cmath>

namespace tune_log10_detail
{
        template
        <
                typename tvalue,
                typename tscalar,
                typename tsize
        >
        void update_range(const tvalue& optimum, const tscalar varlog, const tsize splits, tscalar& minlog, tscalar& maxlog)
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
                typename tscalar,
                typename tsize
        >
        auto evaluate(const toperator& op, const tscalar minlog, const tsize i, const tscalar varlog)
        {
                const auto log = minlog + static_cast<tscalar>(i) * varlog;
                return std::make_tuple(op(make_param(log)), log);
        }
}
