#include "function_trid.h"
#include "math/numeric.hpp"
#include "text/to_string.hpp"

namespace ncv
{
        struct function_trid_t : public function_t
        {
                explicit function_trid_t(const opt_size_t dims)
                        :       m_dims(dims)
                {
                }

                virtual string_t name() const override
                {
                        return "Trid" + text::to_string(m_dims) + "D";
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
                                for (opt_size_t i = 0; i < m_dims; i ++)
                                {
                                        fx += math::square(x(i) - 1.0);
                                }
                                for (opt_size_t i = 1; i < m_dims; i ++)
                                {
                                        fx -= x(i) * x(i - 1);
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const opt_vector_t& x, opt_vector_t& gx)
                        {
                                gx.resize(m_dims);
                                gx.setZero();
                                for (opt_size_t i = 0; i < m_dims; i ++)
                                {
                                        gx(i) += 2.0 * (x(i) - 1.0);
                                }
                                for (opt_size_t i = 1; i < m_dims; i ++)
                                {
                                        gx(i) -= x(i - 1);
                                        gx(i - 1) -= x(i);
                                }

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const opt_vector_t& x) const override
                {
                        return norm(x) < m_dims * m_dims;
                }

                virtual bool is_minima(const opt_vector_t& x, const opt_scalar_t epsilon) const override
                {
                        opt_vector_t xmin(m_dims);
                        for (opt_size_t d = 0; d < m_dims; d ++)
                        {
                                xmin(d) = (d + 1.0) * (m_dims - d);
                        }

                        return distance(x, xmin) < epsilon;
                }

                opt_size_t      m_dims;
        };

        functions_t make_trid_funcs(opt_size_t max_dims)
        {
                functions_t functions;

                for (opt_size_t dims = 2; dims <= max_dims; dims *= 2)
                {
                        functions.push_back(std::make_shared<function_trid_t>(dims));
                }

                return functions;
        }
}
	
