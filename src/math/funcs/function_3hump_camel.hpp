#pragma once

#include "util.hpp"
#include "function.hpp"
#include <cmath>

namespace nano
{
        ///
        /// \brief create three-hump camel test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template
        <
                typename tscalar
        >
        struct function_3hump_camel_t : public function_t<tscalar>
        {
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;

                virtual std::string name() const override
                {
                        return "3hump camel";
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

                                const auto a2 = a * a;
                                const auto a4 = a2 * a2;
                                const auto a6 = a4 * a2;

                                return tscalar(2) * a2 - tscalar(1.05) * a4 + a6 / tscalar(6.0) + a * b + b * b;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const auto a = x(0), b = x(1);

                                const auto a2 = a * a;
                                const auto a3 = a * a2;
                                const auto a5 = a3 * a2;

                                gx.resize(2);
                                gx(0) = tscalar(4) * a - tscalar(1.05) * tscalar(4) * a3 + a5 + b;
                                gx(1) = a + tscalar(2) * b;

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return util::norm(x) < tscalar(5.0);
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        const auto a = tscalar(4.2);
                        const auto b = std::sqrt(tscalar(3.64));

                        const auto xmp = std::sqrt(tscalar(0.5) * (a + b));
                        const auto xmn = std::sqrt(tscalar(0.5) * (a - b));

                        const auto xmins =
                        {
                                std::vector<tscalar>{ tscalar(0.0), tscalar(0.0) },
                                std::vector<tscalar>{ xmp, tscalar(-0.5) * xmp },
                                std::vector<tscalar>{ xmn, tscalar(-0.5) * xmn },
                                std::vector<tscalar>{ -xmp, tscalar(0.5) * xmp },
                                std::vector<tscalar>{ -xmn, tscalar(0.5) * xmn }
                        };

                        return util::check_close(x, xmins, epsilon);
                }
        };
}
