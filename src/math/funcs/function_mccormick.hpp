#pragma once

#include "util.hpp"
#include "function.hpp"
#include <cmath>

namespace math
{
        ///
        /// \brief create McCormick test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template 
        <
                typename tscalar
        >
        struct function_mccormick_t : public function_t<tscalar>
        {
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;

                virtual std::string name() const override
                {
                        return "McCormick";
                }

                virtual tproblem problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return 2;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                const auto a = x(0), b = x(1);

                                return std::sin(a + b) + (a - b) * (a - b) - tscalar(1.5) * a + tscalar(2.5) * b + 1;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const auto a = x(0), b = x(1);

                                gx.resize(2);
                                gx(0) = std::cos(a + b) + 2 * (a - b) - tscalar(1.5);
                                gx(1) = std::cos(a + b) - 2 * (a - b) + tscalar(2.5);

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return  tscalar(-1.5) < x(0) && x(0) < tscalar(4.0) &&
                                tscalar(-3.0) < x(1) && x(1) < tscalar(4.0);
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        const auto xmins =
                        {
                                std::vector<tscalar>{ tscalar(-0.54719), tscalar(-1.54719) }
                        };

                        return util::check_close(x, xmins, epsilon);
                }
        };
}
