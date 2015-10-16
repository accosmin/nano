#pragma once

#include "function.hpp"
#include "tensor/vector.hpp"
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
                typename tscalar
        >
        struct function_3hump_camel_t : public function_t<tscalar>
        {
                typedef typename function_t<tscalar>::tsize     tsize;
                typedef typename function_t<tscalar>::tvector   tvector;
                typedef typename function_t<tscalar>::tproblem  tproblem;  
                
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

                virtual bool is_valid(const tvector& x) const override
                {
                        return norm(x) < 5.0;
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

                        for (const auto& xmin : xmins)
                        {
                                if (distance(x, tensor::map_vector(xmin.data(), 2)) < epsilon)
                                {
                                        return true;
                                }
                        }

                        return false;
                }
        };
}
