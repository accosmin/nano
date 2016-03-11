#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create Beale test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template
        <
                typename tscalar
        >
        struct function_beale_t : public function_t<tscalar>
        {
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;

                virtual std::string name() const override
                {
                        return "Beale";
                }

                virtual tproblem problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return 2;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                const auto a = x(0);
                                const auto b = x(1), b2 = b * b, b3 = b2 * b;

                                const auto z0 = tscalar(1.5) - a + a * b;
                                const auto z1 = tscalar(2.25) - a + a * b2;
                                const auto z2 = tscalar(2.625) - a + a * b3;

                                return z0 * z0 + z1 * z1 + z2 * z2;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const auto a = x(0);
                                const auto b = x(1), b2 = b * b, b3 = b2 * b;

                                const auto z0 = tscalar(1.5) - a + a * b;
                                const auto z1 = tscalar(2.25) - a + a * b2;
                                const auto z2 = tscalar(2.625) - a + a * b3;

                                gx.resize(2);
                                gx(0) = 2 * (z0 * (-1 + b) + z1 * (-1 + b2) + z2 * (-1 + b3));
                                gx(1) = 2 * (z0 * (a) + z1 * (2 * a * b) + z2 * (3 * a * b2));

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return util::norm(x) < tscalar(4.5);
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        const auto xmins =
                        {
                                std::vector<tscalar>{ tscalar(3.0), tscalar(0.5) }
                        };

                        return util::check_close(x, xmins, epsilon);
                }
        };
}
