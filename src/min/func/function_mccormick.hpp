#pragma once

#include "util.hpp"
#include <cmath>

namespace func
{
        ///
        /// \brief create McCormick test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template 
        <
                typename tscalar_
        >
        struct function_mccormick_t
        {
                typedef min::problem_t<tscalar_>        tproblem;
                typedef typename tproblem::tsize        tsize;
                typedef typename tproblem::tscalar      tscalar;
                typedef typename tproblem::tvector      tvector;
                
                std::string name() const
                {
                        return "McCormick";
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

                                return sin(a + b) + (a - b) * (a - b) - 1.5 * a + 2.5 * b + 1;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const tscalar a = x(0), b = x(1);

                                gx.resize(2);
                                gx(0) = cos(a + b) + 2 * (a - b) - 1.5;
                                gx(1) = cos(a + b) - 2 * (a - b) + 2.5;

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                bool is_valid(const tvector& x) const
                {
                        return  -1.5 < x(0) && x(0) < 4.0 &&
                                -3.0 < x(1) && x(1) < 4.0;
                }

                bool is_minima(const tvector& x, const tscalar epsilon) const
                {
                        const auto xmins =
                        {
                                std::vector<tscalar>{ -0.54719, -1.54719 }
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
