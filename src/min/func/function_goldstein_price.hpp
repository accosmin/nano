#pragma once

#include "util.hpp"
#include "function.hpp"

namespace min
{
        ///
        /// \brief create Goldstein-Price test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template
        <
                typename tscalar
        >
        struct function_goldstein_price_t : public function_t<tscalar>
        {
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;

                virtual std::string name() const override
                {
                        return "Goldstein-Price";
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

                                const auto z0 = 1 + a + b;
                                const auto z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
                                const auto z2 = 2 * a - 3 * b;
                                const auto z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

                                const auto u = 1 + z0 * z0 * z1;
                                const auto v = 30 + z2 * z2 * z3;

                                return u * v;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const auto a = x(0), b = x(1);

                                const auto z0 = 1 + a + b;
                                const auto z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
                                const auto z2 = 2 * a - 3 * b;
                                const auto z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

                                const auto u = 1 + z0 * z0 * z1;
                                const auto v = 30 + z2 * z2 * z3;

                                const auto z0da = 1;
                                const auto z0db = 1;

                                const auto z1da = -14 + 6 * a + 6 * b;
                                const auto z1db = -14 + 6 * a + 6 * b;

                                const auto z2da = +2;
                                const auto z2db = -3;

                                const auto z3da = -32 + 24 * a - 36 * b;
                                const auto z3db = +48 - 36 * a + 54 * b;

                                gx.resize(2);
                                gx(0) = u * z2 * (2 * z2da * z3 + z2 * z3da) +
                                        v * z0 * (2 * z0da * z1 + z0 * z1da);
                                gx(1) = u * z2 * (2 * z2db * z3 + z2 * z3db) +
                                        v * z0 * (2 * z0db * z1 + z0 * z1db);

                                return u * v;
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return util::norm(x) < 2.0;
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        const auto xmins =
                        {
                                std::vector<tscalar>{ +0.0, -1.0 },
                                std::vector<tscalar>{ +1.2, +0.8 },
                                std::vector<tscalar>{ +1.8, +0.2 },
                                std::vector<tscalar>{ -0.6, -0.4 }
                        };

                        return util::check_close(x, xmins, epsilon);
                }
        };
}
