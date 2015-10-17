#pragma once

#include "util.hpp"

namespace func
{
        ///
        /// \brief create Beale test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template
        <
                typename tscalar_
        >
        struct function_beale_t
        {
                typedef min::problem_t<tscalar_>        tproblem;
                typedef typename tproblem::tsize        tsize;
                typedef typename tproblem::tscalar      tscalar;
                typedef typename tproblem::tvector      tvector;
                
                std::string name() const
                {
                        return "Beale";
                }

                tproblem problem() const
                {
                        const auto fn_size = [=] ()
                        {
                                return 2;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                const tscalar a = x(0);
                                const tscalar b = x(1), b2 = b * b, b3 = b2 * b;

                                const tscalar z0 = 1.5 - a + a * b;
                                const tscalar z1 = 2.25 - a + a * b2;
                                const tscalar z2 = 2.625 - a + a * b3;

                                return z0 * z0 + z1 * z1 + z2 * z2;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const tscalar a = x(0);
                                const tscalar b = x(1), b2 = b * b, b3 = b2 * b;

                                const tscalar z0 = 1.5 - a + a * b;
                                const tscalar z1 = 2.25 - a + a * b2;
                                const tscalar z2 = 2.625 - a + a * b3;

                                gx.resize(2);
                                gx(0) = 2.0 * (z0 * (-1 + b) + z1 * (-1 + b2) + z2 * (-1 + b3));
                                gx(1) = 2.0 * (z0 * (a) + z1 * (2 * a * b) + z2 * (3 * a * b2));

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                bool is_valid(const tvector& x) const
                {
                        return util::norm(x) < 4.5;
                }

                bool is_minima(const tvector& x, const tscalar epsilon) const
                {
                        const auto xmins =
                        {
                                std::vector<tscalar>{ 3.0, 0.5 }
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
