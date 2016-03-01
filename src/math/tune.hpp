#pragma once

#include <set>
#include "tune_result.hpp"
#include "tune_grid_space.hpp"
#include "tune_finite_space.hpp"

namespace math
{
        ///
        /// \brief search the parameter that minimizes a given operator,
        ///     using the given search space of values to try.
        ///
        template
        <
                typename toperator,     ///< toperator(param) evaluates the parameter
                typename tspace
        >
        auto tune(const toperator& op, tspace space)
        {
                using toptimum = decltype(op(typename tspace::tscalar(0)));
                using tparameters = std::tuple<typename tspace::tscalar>;

                tune_result_t<toptimum, tparameters> result;
                do
                {
                        for (const auto param : space.values())
                        {
                                result.update(op(param), std::make_tuple(param));
                        }
                }
                while (space.refine(result.template param<0>()));

                return result;
        }

        ///
        /// \brief search the parameters that minimizes a given operator,
        ///     using the given search spaces of values to try.
        ///
        template
        <
                typename toperator,     ///< toperator(param, paramX...) evaluates the parameters
                typename tspace,
                typename... tspaceX
        >
        auto tune(const toperator& op, tspace space, tspaceX... spaceX)
        {
                using toptimum = decltype(op(typename tspace::tscalar(0), typename tspaceX::tscalar(0)...));
                using tparameters = std::tuple<typename tspace::tscalar, typename tspaceX::tscalar...>;

                tune_result_t<toptimum, tparameters> result;
                do
                {
                        for (const auto param : space.values())
                        {
                                const auto opp = [&] (const auto... paramX)
                                {
                                        return op(param, paramX...);
                                };
                                const auto trial = tune(opp, spaceX...);
                                result.update(trial.optimum(), std::tuple_cat(std::tie(param), trial.params()));
                        }
                }
                while (space.refine(result.template param<0>()));

                return result;
        }
}
