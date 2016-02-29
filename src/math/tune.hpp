#pragma once

#include <set>
#include "tune_grid_space.hpp"
#include "tune_finite_space.hpp"

namespace math
{
        ///
        /// \brief search the parameter that minimizes a given operator, using a search space of values to try.
        ///
        /// \returns { the result associated to the optimum parameter, the optimum parameter }.
        ///
        template
        <
                typename toperator,     ///< toperator(param) evaluates the parameter
                typename tspace
        >
        auto tune(const toperator& op, tspace space)
        {
                using tvalue = typename tspace::tscalar;
                using tresult = decltype(op(tvalue(0)));
                using trecord = std::tuple<tresult, tvalue>;

                std::set<trecord> history;
                do
                {
                        for (const auto param : space.values())
                        {
                                history.emplace(op(param), param);
                        }
                        assert(!history.empty());
                }
                while (space.refine(std::get<1>(*history.begin())));

                return *history.begin();
        }

        template <typename tfirst, typename... trest>
        std::tuple<trest...> remove_first(const std::tuple<tfirst, trest...>& tuple);

        ///
        /// \brief search the parameters that minimizes a given operator, using a set of search spaces of values to try.
        ///
        /// \returns { the result associated to the optimum parameter, the optimum parameters }.
        ///
        template
        <
                typename toperator,     ///< toperator(param, paramX...) evaluates the parameters
                typename tspace,
                typename... tspaceX
        >
        auto tune(const toperator& op, tspace space, tspaceX... spaceX)
        {
                using tvalue = typename tspace::tscalar;
                using tresult = decltype(op(tvalue(0), typename tspaceX::tscalar(0)...));
                using trecord = std::tuple<tresult, tvalue, typename tspaceX::tscalar...>;

                std::set<trecord> history;
                do
                {
                        for (const auto param : space.values())
                        {
                                const auto opp = [&] (const auto... paramX)
                                {
                                        return op(param, paramX...);
                                };
                                const auto trial = tune(opp, spaceX...);
                                history.insert(std::tuple_cat(std::tie(std::get<0>(trial)), std::tie(param), remove_first(trial)));
                        }
                        assert(!history.empty());
                }
                while (space.refine(std::get<1>(*history.begin())));

                return *history.begin();
        }
}
