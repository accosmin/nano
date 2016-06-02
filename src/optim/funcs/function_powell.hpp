#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create Powell test functions
        ///
        struct function_powell_t : public function_t
        {
                explicit function_powell_t(const tsize dims) :
                        m_dims(std::max(tsize(4), dims - dims % 4))
                {
                }

                virtual std::string name() const override
                {
                        return "Powell" + std::to_string(m_dims) + "D";
                }

                virtual problem_t problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const vector_t& x)
                        {
                                scalar_t fx = 0;
                                for (tsize i = 0, i4 = 0; i < m_dims / 4; i ++, i4 += 4)
                                {
                                        fx += nano::square(x(i4 + 0) + x(i4 + 1) * 10);
                                        fx += nano::square(x(i4 + 2) - x(i4 + 3)) * 5;
                                        fx += nano::quartic(x(i4 + 1) - x(i4 + 2) * 2);
                                        fx += nano::quartic(x(i4 + 0) - x(i4 + 3)) * 10;
                                }

                                return fx;
                        };

                        const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx.resize(m_dims);
                                for (tsize i = 0, i4 = 0; i < m_dims / 4; i ++, i4 += 4)
                                {
                                        const auto gfx1 = (x(i4 + 0) + x(i4 + 1) * 10) * 2;
                                        const auto gfx2 = (x(i4 + 2) - x(i4 + 3)) * 5 * 2;
                                        const auto gfx3 = nano::cube(x(i4 + 1) - x(i4 + 2) * 2) * 4;
                                        const auto gfx4 = nano::cube(x(i4 + 0) - x(i4 + 3)) * 10 * 4;

                                        gx(i4 + 0) = gfx1 + gfx4;
                                        gx(i4 + 1) = gfx1 * 10 + gfx3;
                                        gx(i4 + 2) = gfx2 - 2 * gfx3;
                                        gx(i4 + 3) = - gfx2 - gfx4;
                                }

                                return fn_fval(x);
                        };

                        return {fn_size, fn_fval, fn_grad};
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return scalar_t(-4) < x.minCoeff() && x.maxCoeff() < scalar_t(5);
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
                }

                tsize   m_dims;
        };
}
