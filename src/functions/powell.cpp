#include "powell.h"
#include "util.hpp"

namespace nano
{
        function_powell_t::function_powell_t(const tensor_size_t dims) :
                m_dims(std::max(tensor_size_t(4), dims - dims % 4))
        {
        }

        std::string function_powell_t::name() const
        {
                return "Powell" + std::to_string(m_dims) + "D";
        }

        problem_t function_powell_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        scalar_t fx = 0;
                        for (tensor_size_t i = 0, i4 = 0; i < m_dims / 4; i ++, i4 += 4)
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
                        for (tensor_size_t i = 0, i4 = 0; i < m_dims / 4; i ++, i4 += 4)
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

        bool function_powell_t::is_valid(const vector_t& x) const
        {
                return scalar_t(-4) < x.minCoeff() && x.maxCoeff() < scalar_t(5);
        }

        bool function_powell_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
        }

        bool function_powell_t::is_convex() const
        {
                return true;
        }

        tensor_size_t function_powell_t::min_dims() const
        {
                return 4;
        }

        tensor_size_t function_powell_t::max_dims() const
        {
                return 100 * 1000;
        }
}
