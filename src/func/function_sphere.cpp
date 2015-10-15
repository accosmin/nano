#include "function_sphere.h"

namespace ncv
{
        struct function_sphere_t : public function_t
        {
                explicit function_sphere_t(const opt_size_t dims)
                        :       m_dims(dims)
                {
                }

                virtual string_t name() const override
                {
                        return "sphere" + std::to_string(m_dims) + "D";
                }

                virtual opt_problem_t problem() const override
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const opt_opfval_t fn_fval = [=] (const opt_vector_t& x)
                        {
                                return x.dot(x);
                        };

                        const opt_opgrad_t fn_grad = [=] (const opt_vector_t& x, opt_vector_t& gx)
                        {
                                gx = 2 * x;

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

        functions_t make_sphere_funcs(opt_size_t max_dims)
        {
                functions_t functions;

                for (opt_size_t dims = 1; dims <= max_dims; dims *= 2)
                {
                        functions.push_back(std::make_shared<function_sphere_t>(dims));
                }

                return functions;
        }
}
	
