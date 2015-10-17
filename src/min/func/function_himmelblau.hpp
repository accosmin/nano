#pragma once

#include "util.hpp"

namespace func
{
        ///
        /// \brief create Himmelblau test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template 
        <
                typename tscalar_
        >
        struct function_himmelblau_t
        {
                typedef min::problem_t<tscalar_>        tproblem;
                typedef typename tproblem::tsize        tsize;
                typedef typename tproblem::tscalar      tscalar;
                typedef typename tproblem::tvector      tvector;
                
                std::string name() const
                {
                        return "Himmelblau";
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

                                const tscalar u = a * a + b - 11;
                                const tscalar v = a + b * b - 7;

                                return u * u + v * v;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const tscalar a = x(0), b = x(1);

                                const tscalar u = a * a + b - 11;
                                const tscalar v = a + b * b - 7;

                                gx.resize(2);
                                gx(0) = 2 * u * 2 * a + 2 * v;
                                gx(1) = 2 * u + 2 * v * 2 * b;

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                bool is_valid(const tvector&) const
                {
                        return true;
                }

                bool is_minima(const tvector& x, const tscalar epsilon) const
                {
                        const auto xmins =
                        {
                                std::vector<tscalar>{ 3.0, 2.0 },
                                std::vector<tscalar>{ -2.805118, 3.131312 },
                                std::vector<tscalar>{ -3.779310, -3.283186 },
                                std::vector<tscalar>{ 3.584428, -1.848126 }
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
