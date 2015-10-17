#pragma once

#include "util.hpp"
#include <cmath>

namespace func
{
        ///
        /// \brief create three-hump camel test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template
        <
                typename tscalar_
        >
        struct function_3hump_camel_t
        {
                typedef min::problem_t<tscalar_>        tproblem;
                typedef typename tproblem::tsize        tsize;
                typedef typename tproblem::tscalar      tscalar;
                typedef typename tproblem::tvector      tvector;
                
                std::string name() const
                {
                        return "3hump camel";
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

                                const tscalar a2 = a * a;
                                const tscalar a4 = a2 * a2;
                                const tscalar a6 = a4 * a2;

                                return 2 * a2 - 1.05 * a4 + a6 / 6.0 + a * b + b * b;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const tscalar a = x(0), b = x(1);

                                const tscalar a2 = a * a;
                                const tscalar a3 = a * a2;
                                const tscalar a5 = a3 * a2;

                                gx.resize(2);
                                gx(0) = 4 * a - 1.05 * 4 * a3 + a5 + b;
                                gx(1) = a + 2 * b;

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                bool is_valid(const tvector& x) const
                {
                        return util::norm(x) < 5.0;
                }

                bool is_minima(const tvector& x, const tscalar epsilon) const
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
