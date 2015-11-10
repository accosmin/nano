#pragma once

#include "util.hpp"
#include "function.hpp"

namespace math
{
        ///
        /// \brief create Himmelblau test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template 
        <
                typename tscalar
        >
        struct function_himmelblau_t : public function_t<tscalar>
        {
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;

                virtual std::string name() const override
                {
                        return "Himmelblau";
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

                                const auto u = a * a + b - 11;
                                const auto v = a + b * b - 7;

                                return u * u + v * v;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const auto a = x(0), b = x(1);

                                const auto u = a * a + b - 11;
                                const auto v = a + b * b - 7;

                                gx.resize(2);
                                gx(0) = 2 * u * 2 * a + 2 * v;
                                gx(1) = 2 * u + 2 * v * 2 * b;

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector&) const override
                {
                        return true;
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        const auto xmins =
                        {
                                std::vector<tscalar>{ tscalar(3.0), tscalar(2.0) },
                                std::vector<tscalar>{ tscalar(-2.805118), tscalar(3.131312) },
                                std::vector<tscalar>{ tscalar(-3.779310), tscalar(-3.283186) },
                                std::vector<tscalar>{ tscalar(3.584428), tscalar(-1.848126) }
                        };

                        return util::check_close(x, xmins, epsilon);
                }
        };
}
