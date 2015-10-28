#pragma once

#include "util.hpp"
#include "function.hpp"

namespace min
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
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;

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
                                        fx += util::square(x(i4 + 0) + x(i4 + 1) * 10);
                                        fx += util::square(x(i4 + 2) - x(i4 + 3)) * 5;
                                        fx += util::quartic(x(i4 + 1) - x(i4 + 2) * 2);
                                        fx += util::quartic(x(i4 + 0) - x(i4 + 3)) * 10;
                                }

                                return fx;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                gx.resize(m_dims);
                                for (tsize i = 0, i4 = 0; i < m_dims / 4; i ++, i4 += 4)
                                {
                                        const auto gfx1 = (x(i4 + 0) + x(i4 + 1) * 10) * 2;
                                        const auto gfx2 = (x(i4 + 2) - x(i4 + 3)) * 5 * 2;
                                        const auto gfx3 = util::cube(x(i4 + 1) - x(i4 + 2) * 2) * 4;
                                        const auto gfx4 = util::cube(x(i4 + 0) - x(i4 + 3)) * 10 * 4;

                                        gx(i4 + 0) = gfx1 + gfx4;
                                        gx(i4 + 1) = gfx1 * 10 + gfx3;
                                        gx(i4 + 2) = gfx2 - 2 * gfx3;
                                        gx(i4 + 3) = - gfx2 - gfx4;
                                }

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return tscalar(-4) < x.minCoeff() && x.maxCoeff() < tscalar(5);
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        return util::distance(x, tvector::Zero(m_dims)) < epsilon;
                }

                tsize   m_dims;
        };
}
