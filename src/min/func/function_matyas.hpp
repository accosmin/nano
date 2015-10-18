#pragma once

#include "function.hpp"

namespace func
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
                typedef typename function_t<tscalar>::tsize     tsize;
                typedef typename function_t<tscalar>::tvector   tvector;
                typedef typename function_t<tscalar>::tproblem  tproblem;                
                
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

                virtual bool is_valid(const tvector& x) const override
                {
                        return util::norm(x) < 10.0;
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        return util::distance(x, tvector::Zero(2)) < epsilon;
                }
        };
}
