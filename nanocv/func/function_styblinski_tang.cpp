#include "function_styblinski_tang.h"
#include "math/numeric.hpp"
#include "text/to_string.hpp"

namespace ncv
{
        struct function_styblinski_tang_t : public function_t
        {
                explicit function_styblinski_tang_t(const opt_size_t dims)
                        :       m_dims(dims)
                {
                }

                virtual string_t name() const override
                {
                        return "Styblinski-Tang" + text::to_string(m_dims) + "D";
                }

                virtual opt_problem_t problem() const override
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const opt_opfval_t fn_fval = [=] (const opt_vector_t& x)
                        {
                                opt_scalar_t u = 0;
                                for (opt_size_t i = 0; i < m_dims; i ++)
                                {
                                        u += math::quartic(x(i)) - 16 * math::square(x(i)) + 5 * x(i);
                                }

                                return 0.5 * u;
                        };

                        const opt_opgrad_t fn_grad = [=] (const opt_vector_t& x, opt_vector_t& gx)
                        {
                                gx.resize(m_dims);
                                for (opt_size_t i = 0; i < m_dims; i ++)
                                {
                                        gx(i) = 2 * math::cube(x(i)) - 16 * x(i) + 2.5;
                                }

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const opt_vector_t& x) const override
                {
                        return -5.0 < x.minCoeff() && x.maxCoeff() < 5.0;
                }

                virtual bool is_minima(const opt_vector_t& x, const opt_scalar_t epsilon) const override
                {
                        return distance(x, opt_vector_t::Constant(-2.903534, m_dims)) < epsilon;
                }

                opt_size_t      m_dims;
        };

        functions_t make_styblinski_tang_funcs(opt_size_t max_dims)
        {
                functions_t functions;

                for (opt_size_t dims = 1; dims <= max_dims; dims *= 2)
                {
                        functions.push_back(std::make_shared<function_styblinski_tang_t>(dims));
                }

                return functions;
        }
}
	
