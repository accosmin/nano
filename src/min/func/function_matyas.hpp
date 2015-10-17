#pragma once

#include "util.hpp"

namespace func
{
        ///
        /// \brief create Matyas test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template 
        <
                typename tscalar_
        >
        struct function_matyas_t
        {
                typedef min::problem_t<tscalar_>        tproblem;
                typedef typename tproblem::tsize        tsize;
                typedef typename tproblem::tscalar      tscalar;
                typedef typename tproblem::tvector      tvector;
                
                std::string name() const
                {
                        return "Matyas";
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

                                return 0.26 * (a * a + b * b) - 0.48 * a * b;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const tscalar a = x(0), b = x(1);

                                gx.resize(2);
                                gx(0) = 0.26 * 2 * a - 0.48 * b;
                                gx(1) = 0.26 * 2 * b - 0.48 * a;

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
                        return util::distance(x, tvector::Zero(2)) < epsilon;
                }
        };
}
