#pragma once

#include "util.hpp"
#include "function.hpp"

namespace min
{
        ///
        /// \brief create Matyas test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template 
        <
                typename tscalar 
        >
        struct function_matyas_t : public function_t<tscalar>
        {
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;

                virtual std::string name() const override
                {
                        return "Matyas";
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

                                return tscalar(0.26) * (a * a + b * b) - tscalar(0.48) * a * b;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const auto a = x(0), b = x(1);

                                gx.resize(2);
                                gx(0) = tscalar(0.26) * 2 * a - tscalar(0.48) * b;
                                gx(1) = tscalar(0.26) * 2 * b - tscalar(0.48) * a;

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return util::norm(x) < tscalar(10);
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        return util::distance(x, tvector::Zero(2)) < epsilon;
                }
        };
}
