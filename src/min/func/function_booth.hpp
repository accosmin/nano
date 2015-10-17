#pragma once

#include "util.hpp"

namespace func
{
        ///
        /// \brief create Booth test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template
        <
                typename tscalar_
        >
        struct function_booth_t
        {
                typedef min::problem_t<tscalar_>        tproblem;
                typedef typename tproblem::tsize        tsize;
                typedef typename tproblem::tscalar      tscalar;
                typedef typename tproblem::tvector      tvector;
                
                std::string name() const
                {
                        return "Booth";
                }

                tproblem problem() const
                {
                        const auto fn_size = [=] ()
                        {
                                return 2;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                const tscalar a = x(0), b = x(1);

                                const tscalar u = a + 2 * b - 7;
                                const tscalar v = 2 * a + b - 5;

                                return u * u + v * v;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const tscalar a = x(0), b = x(1);

                                const tscalar u = a + 2 * b - 7;
                                const tscalar v = 2 * a + b - 5;

                                gx.resize(2);
                                gx(0) = 2 * u + 2 * v * 2;
                                gx(1) = 2 * u * 2 + 2 * v;

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                bool is_valid(const tvector& x) const
                {
                        return util::norm(x) < 10.0;
                }

                bool is_minima(const tvector& x, const tscalar epsilon) const
                {
                        const auto xmins =
                        {
                                std::vector<tscalar>{ 1.0, 3.0 }
                        };

                        for (const auto& xmin : xmins)
                        {
                                if (util::distance(x, util::map_vector(xmin.data(), 2)) < epsilon)
                                {
                                        return true;
                                }
                        }

                        return false;
                }
        };
}
