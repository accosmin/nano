#include "function_sum_squares.h"

namespace ncv
{
        struct function_sum_squares_t : public function_t
        {
                explicit function_sum_squares_t(const opt_size_t dims)
                        :       m_dims(dims)
                {
                }

                virtual string_t name() const override
                {
                        return "sum squares" + std::to_string(m_dims) + "D";
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
                                        fx += (i + 1) * x(i) * x(i);
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const opt_vector_t& x, opt_vector_t& gx)
                        {
                                gx.resize(m_dims);
                                for (opt_size_t i = 0; i < m_dims; i ++)
                                {
                                        gx(i) = 2.0 * (i + 1) * x(i);
                                }

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const opt_vector_t& x) const override
                {
                        return norm(x) < 5.12;
                }

                virtual bool is_minima(const opt_vector_t& x, const opt_scalar_t epsilon) const override
                {
                        return distance(x, opt_vector_t::Zero(m_dims)) < epsilon;
                }

                opt_size_t      m_dims;
        };

        functions_t make_sum_squares_funcs(opt_size_t max_dims)
        {
                functions_t functions;

                for (opt_size_t dims = 2; dims <= max_dims; dims *= 2)
                {
                        functions.push_back(std::make_shared<function_sum_squares_t>(dims));
                }

                return functions;
        }
}
	
