#include "function_sphere.h"
#include "text/to_string.hpp"

namespace ncv
{
        struct function_sphere_t : public function_t
        {
                explicit function_sphere_t(const size_t dims)
                        :       m_dims(dims)
                {
                }

                virtual string_t name() const override
                {
                        return "sphere" + text::to_string(m_dims) + "D";
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
                                for (size_t i = 0; i < m_dims; i ++)
                                {
                                        fx += x(i) * x(i);
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx.resize(m_dims);
                                for (size_t i = 0; i < m_dims; i ++)
                                {
                                        gx(i) = 2 * x(i);
                                }

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return x.lpNorm<Eigen::Infinity>() < 5.12;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        return (x - vector_t::Zero(m_dims)).lpNorm<Eigen::Infinity>() < epsilon;
                }

                size_t  m_dims;
        };

        functions_t make_sphere_funcs(size_t max_dims)
        {
                functions_t functions;

                for (size_t dims = 1; dims <= max_dims; dims *= 2)
                {
                        functions.push_back(std::make_shared<function_sphere_t>(dims));
                }

                return functions;
        }
}
	
