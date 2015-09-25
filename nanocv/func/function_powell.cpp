#include "function_powell.h"
#include "math/numeric.hpp"
#include "text/to_string.hpp"

namespace ncv
{
        struct function_powell_t : public function_t
        {
                explicit function_powell_t(const size_t dims)
                        :       m_dims(dims)
                {
                }

                virtual string_t name() const override
                {
                        return "Powell" + text::to_string(m_dims) + "D";
                }

                virtual opt_problem_t problem() const override
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const opt_opfval_t fn_fval = [=] (const vector_t& x)
                        {
                                scalar_t fx = 0;
                                for (size_t i = 0, i4 = 0; i < m_dims / 4; i ++, i4 += 4)
                                {
                                        fx += math::square(x(i4 + 0) + x(i4 + 1) * 10.0);
                                        fx += math::square(x(i4 + 2) - x(i4 + 3)) * 5.0;
                                        fx += math::quartic(x(i4 + 1) - x(i4 + 2) * 2.0);
                                        fx += math::quartic(x(i4 + 0) - x(i4 + 3)) * 10.0;
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx.resize(m_dims);
                                gx.setZero();
                                for (size_t i = 0, i4 = 0; i < m_dims / 4; i ++, i4 += 4)
                                {
                                        const scalar_t gfx1 = (x(i4 + 0) + x(i4 + 1) * 10.0) * 2.0;
                                        const scalar_t gfx2 = (x(i4 + 2) - x(i4 + 3)) * 5.0 * 2.0;
                                        const scalar_t gfx3 = math::cube(x(i4 + 1) - x(i4 + 2) * 2.0) * 4.0;
                                        const scalar_t gfx4 = math::cube(x(i4 + 0) - x(i4 + 3)) * 10.0 * 4.0;

                                        gx(i4 + 0) += gfx1 + gfx4;
                                        gx(i4 + 1) += gfx1 * 10.0 + gfx3;
                                        gx(i4 + 2) += gfx2 - 2.0 * gfx3;
                                        gx(i4 + 3) += - gfx2 - gfx4;
                                }

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return -4.0 < x.minCoeff() && x.maxCoeff() < 5.0;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        return distance(x, vector_t::Zero(m_dims)) < epsilon;
                }

                size_t  m_dims;
        };

        functions_t make_powell_funcs(size_t max_dims)
        {
                functions_t functions;

                for (size_t dims = 4; dims <= max_dims; dims *= 4)
                {
                        functions.push_back(std::make_shared<function_powell_t>(dims));
                }

                return functions;
        }
}
	
