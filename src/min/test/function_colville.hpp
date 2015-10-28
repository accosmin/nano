#pragma once

#include "util.hpp"
#include "function.hpp"

namespace min
{
        ///
        /// \brief create Colville test functions
        ///
        template
        <
                typename tscalar
        >
        struct function_colville_t : public function_t<tscalar>
        {
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;

                explicit function_colville_t()
                {
                }

                virtual std::string name() const override
                {
                        return "Colville";
                }

                virtual tproblem problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return 4;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                const auto x1 = x(0);
                                const auto x2 = x(1);
                                const auto x3 = x(2);
                                const auto x4 = x(3);

                                return  100 * util::square(x1 * x1 - x2) +
                                        util::square(x1 - 1) +
                                        util::square(x3 - 1) +
                                        90 * util::square(x3 * x3 - x4) +
                                        tscalar(10.1) * util::square(x2 - 1) +
                                        tscalar(10.1) * util::square(x4 - 1) +
                                        tscalar(19.8) * (x2 - 1) * (x4 - 1);
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const auto x1 = x(0);
                                const auto x2 = x(1);
                                const auto x3 = x(2);
                                const auto x4 = x(3);

                                gx.resize(4);
                                gx(0) = 400 * (x1 * x1 - x2) * x1 + 2 * (x1 - 1);
                                gx(1) = -200 * (x1 * x1 - x2) + tscalar(20.2) * (x2 - 1) + tscalar(19.8) * (x4 - 1);
                                gx(2) = 360 * (x3 * x3 - x4) * x3 + 2 * (x3 - 1);
                                gx(3) = -180 * (x3 * x3 - x4) + tscalar(20.2) * (x4 - 1) + tscalar(19.8) * (x2 - 1);

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return -10.0 < x.minCoeff() && x.maxCoeff() < tscalar(10);
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        return util::distance(x, tvector::Ones(4)) < epsilon;
                }
        };
}
