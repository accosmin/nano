#pragma once

#include "function.hpp"
#include "math/numeric.hpp"

namespace func
{
        ///
        /// \brief create Powell test functions
        ///
        template
        <
                typename tscalar
        >
        struct function_powell_t : public function_t<tscalar>
        {
                typedef typename function_t<tscalar>::tsize     tsize;
                typedef typename function_t<tscalar>::tvector   tvector;
                typedef typename function_t<tscalar>::tproblem  tproblem;  
                
                explicit function_powell_t(const tsize dims)
                        :       m_dims(std::max(tsize(4), dims - dims % 4))
                {
                }

                virtual std::string name() const override
                {
                        return "Powell" + std::to_string(m_dims) + "D";
                }

                virtual tproblem problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                tscalar fx = 0;
                                for (tsize i = 0, i4 = 0; i < m_dims / 4; i ++, i4 += 4)
                                {
                                        fx += math::square(x(i4 + 0) + x(i4 + 1) * 10.0);
                                        fx += math::square(x(i4 + 2) - x(i4 + 3)) * 5.0;
                                        fx += math::quartic(x(i4 + 1) - x(i4 + 2) * 2.0);
                                        fx += math::quartic(x(i4 + 0) - x(i4 + 3)) * 10.0;
                                }

                                return fx;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                gx.resize(m_dims);
                                for (tsize i = 0, i4 = 0; i < m_dims / 4; i ++, i4 += 4)
                                {
                                        const tscalar gfx1 = (x(i4 + 0) + x(i4 + 1) * 10.0) * 2.0;
                                        const tscalar gfx2 = (x(i4 + 2) - x(i4 + 3)) * 5.0 * 2.0;
                                        const tscalar gfx3 = math::cube(x(i4 + 1) - x(i4 + 2) * 2.0) * 4.0;
                                        const tscalar gfx4 = math::cube(x(i4 + 0) - x(i4 + 3)) * 10.0 * 4.0;

                                        gx(i4 + 0) = gfx1 + gfx4;
                                        gx(i4 + 1) = gfx1 * 10.0 + gfx3;
                                        gx(i4 + 2) = gfx2 - 2.0 * gfx3;
                                        gx(i4 + 3) = - gfx2 - gfx4;
                                }

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return -4.0 < x.minCoeff() && x.maxCoeff() < 5.0;
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        return distance(x, tvector::Zero(m_dims)) < epsilon;
                }

                tsize   m_dims;
        };
}
