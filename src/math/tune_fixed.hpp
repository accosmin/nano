#pragma once

#include <set>
#include <tuple>
#include <cassert>
#include <type_traits>

namespace math
{
        ///
        /// \brief search for a 1D parameter that minimizes a given operator, using a fixed set of values to try.
        ///
        /// \returns { the result associated to the optimum parameter, the optimum parameter }.
        ///
        template
        <
                typename toperator,     ///< toperator(tvalue1) evaluates the parameter
                typename tvalues1
        >
        auto tune_fixed(const toperator& op,
                const tvalues1& values1)
        {
                using tvalue1 = typename std::remove_reference<decltype(*values1.begin())>::type;
                using tresult = decltype(op(tvalue1(0)));
                using trecord = std::tuple<tresult, tvalue1>;

                std::set<trecord> history;
                for (auto param1 : values1)
                {
                        history.emplace(op(param1), param1);
                }

                assert(!history.empty());
                return *history.begin();
        }

        ///
        /// \brief search for a 2D parameter that minimizes a given operator, using a fixed set of values to try.
        ///
        /// \returns { the result associated to the optimum parameter, the optimum parameters }.
        ///
        template
        <
                typename toperator,     ///< toperator(tvalue1, tvalue2) evaluates the parameters
                typename tvalues1,
                typename tvalues2
        >
        auto tune_fixed(const toperator& op,
                const tvalues1& values1, const tvalues2& values2)
        {
                using tvalue1 = typename std::remove_reference<decltype(*values1.begin())>::type;
                using tvalue2 = typename std::remove_reference<decltype(*values2.begin())>::type;
                using tresult = decltype(op(tvalue1(0), tvalue2(0)));
                using trecord = std::tuple<tresult, tvalue1, tvalue2>;

                std::set<trecord> history;
                for (auto param2 : values2)
                {
                        const auto opp = [&] (const auto param1)
                        {
                                return op(param1, param2);
                        };
                        history.insert(std::tuple_cat(tune_fixed(opp, values1), std::tie(param2)));
                }

                assert(!history.empty());
                return *history.begin();
        }

        ///
        /// \brief search for a 3D parameter that minimizes a given operator, using a fixed set of values to try.
        ///
        /// \returns { the result associated to the optimum parameter, the optimum parameters }.
        ///
        template
        <
                typename toperator,     ///< toperator(tvalue1, tvalue2, tvalue3) evaluates the parameters
                typename tvalues1,
                typename tvalues2,
                typename tvalues3
        >
        auto tune_fixed(const toperator& op,
                const tvalues1& values1, const tvalues2& values2, const tvalues3& values3)
        {
                using tvalue1 = typename std::remove_reference<decltype(*values1.begin())>::type;
                using tvalue2 = typename std::remove_reference<decltype(*values2.begin())>::type;
                using tvalue3 = typename std::remove_reference<decltype(*values3.begin())>::type;
                using tresult = decltype(op(tvalue1(0), tvalue2(0), tvalue3(3)));
                using trecord = std::tuple<tresult, tvalue1, tvalue2, tvalue3>;

                std::set<trecord> history;
                for (auto param3 : values3)
                {
                        const auto opp = [&] (const auto param1, const auto param2)
                        {
                                return op(param1, param2, param3);
                        };
                        history.insert(std::tuple_cat(tune_fixed(opp, values1, values2), std::tie(param3)));
                }

                assert(!history.empty());
                return *history.begin();
        }

        ///
        /// \brief search for a 4D parameter that minimizes a given operator, using a fixed set of values to try.
        ///
        /// \returns { the result associated to the optimum parameter, the optimum parameters }.
        ///
        template
        <
                typename toperator,     ///< toperator(tvalue1, tvalue2, tvalue3, tvalue4) evaluates the parameters
                typename tvalues1,
                typename tvalues2,
                typename tvalues3,
                typename tvalues4
        >
        auto tune_fixed(const toperator& op,
                const tvalues1& values1, const tvalues2& values2, const tvalues3& values3, const tvalues4& values4)
        {
                using tvalue1 = typename std::remove_reference<decltype(*values1.begin())>::type;
                using tvalue2 = typename std::remove_reference<decltype(*values2.begin())>::type;
                using tvalue3 = typename std::remove_reference<decltype(*values3.begin())>::type;
                using tvalue4 = typename std::remove_reference<decltype(*values4.begin())>::type;
                using tresult = decltype(op(tvalue1(0), tvalue2(0), tvalue3(0), tvalue4(0)));
                using trecord = std::tuple<tresult, tvalue1, tvalue2, tvalue3, tvalue4>;

                std::set<trecord> history;
                for (auto param4 : values4)
                {
                        const auto opp = [&] (const auto param1, const auto param2, const auto param3)
                        {
                                return op(param1, param2, param3, param4);
                        };
                        history.insert(std::tuple_cat(tune_fixed(opp, values1, values2, values3), std::tie(param4)));
                }

                assert(!history.empty());
                return *history.begin();
        }
}
