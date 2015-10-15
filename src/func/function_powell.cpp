#include "function_powell.h"
#include "math/numeric.hpp"

namespace ncv
{
        struct function_powell_t : public function_t
        {
                explicit function_powell_t(const opt_size_t dims)
                        :       m_dims(dims)
                {
                }

                virtual string_t name() const override
                {
                        return "Powell" + std::to_string(m_dims) + "D";
                }

                virtual opt_problem_t problem() const override
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const opt_opfval_t fn_fval = [=] (const opt_vector_t& x)
                        {
                                opt_scalar_t fx = 0;
                                for (opt_size_t i = 0, i4 = 0; i < m_dims / 4; i ++, i4 += 4)
                                {
                                        fx += math::square(x(i4 + 0) + x(i4 + 1) * 10.0);
                                        fx += math::square(x(i4 + 2) - x(i4 + 3)) * 5.0;
                                        fx += math::quartic(x(i4 + 1) - x(i4 + 2) * 2.0);
                                        fx += math::quartic(x(i4 + 0) - x(i4 + 3)) * 10.0;
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const opt_vector_t& x, opt_vector_t& gx)
                        {
                                gx.resize(m_dims);
                                for (opt_size_t i = 0, i4 = 0; i < m_dims / 4; i ++, i4 += 4)
                                {
                                        const opt_scalar_t gfx1 = (x(i4 + 0) + x(i4 + 1) * 10.0) * 2.0;
                                        const opt_scalar_t gfx2 = (x(i4 + 2) - x(i4 + 3)) * 5.0 * 2.0;
                                        const opt_scalar_t gfx3 = math::cube(x(i4 + 1) - x(i4 + 2) * 2.0) * 4.0;
                                        const opt_scalar_t gfx4 = math::cube(x(i4 + 0) - x(i4 + 3)) * 10.0 * 4.0;

                                        gx(i4 + 0) = gfx1 + gfx4;
                                        gx(i4 + 1) = gfx1 * 10.0 + gfx3;
                                        gx(i4 + 2) = gfx2 - 2.0 * gfx3;
                                        gx(i4 + 3) = - gfx2 - gfx4;
                                }

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const opt_vector_t& x) const override
                {
                        return -4.0 < x.minCoeff() && x.maxCoeff() < 5.0;
                }

                virtual bool is_minima(const opt_vector_t& x, const opt_scalar_t epsilon) const override
                {
                        return distance(x, opt_vector_t::Zero(m_dims)) < epsilon;
                }

                opt_size_t      m_dims;
        };

        functions_t make_powell_funcs(opt_size_t max_dims)
        {
                functions_t functions;

                for (opt_size_t dims = 4; dims <= max_dims; dims *= 4)
                {
                        functions.push_back(std::make_shared<function_powell_t>(dims));
                }

                return functions;
        }
}
	
