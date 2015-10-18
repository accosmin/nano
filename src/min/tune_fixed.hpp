#pragma once

#include <set>
#include <tuple>
#include <type_traits>

namespace min
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
                typedef typename std::remove_reference<decltype(*values1.begin())>::type        tvalue1;
                typedef decltype(op(tvalue1(0)))                                                tresult;
                typedef std::tuple<tresult, tvalue1>                                            trecord;

                std::set<trecord> history;
                for (auto param1 : values1)
                {
                        history.emplace(op(param1), param1);
                }

                return history.empty() ? trecord() : *history.begin();
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
                typedef typename std::remove_reference<decltype(*values1.begin())>::type        tvalue1;
                typedef typename std::remove_reference<decltype(*values2.begin())>::type        tvalue2;
                typedef decltype(op(tvalue1(0), tvalue2(0)))                                    tresult;
                typedef std::tuple<tresult, tvalue1, tvalue2>                                   trecord;

                std::set<trecord> history;
                for (auto param1 : values1)
                {
                        for (auto param2 : values2)
                        {
                                history.emplace(op(param1, param2), param1, param2);
                        }
                }

                return history.empty() ? trecord() : *history.begin();
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
                typedef typename std::remove_reference<decltype(*values1.begin())>::type        tvalue1;
                typedef typename std::remove_reference<decltype(*values2.begin())>::type        tvalue2;
                typedef typename std::remove_reference<decltype(*values3.begin())>::type        tvalue3;
                typedef decltype(op(tvalue1(0), tvalue2(0), tvalue3(0)))                        tresult;
                typedef std::tuple<tresult, tvalue1, tvalue2, tvalue3>                          trecord;

                std::set<trecord> history;
                for (auto param1 : values1)
                {
                        for (auto param2 : values2)
                        {
                                for (auto param3 : values3)
                                {
                                        history.emplace(op(param1, param2, param3), param1, param2, param3);
                                }
                        }
                }

                return history.empty() ? trecord() : *history.begin();
        }
}
