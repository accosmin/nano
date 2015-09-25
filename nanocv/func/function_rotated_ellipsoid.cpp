#include "function_rotated_ellipsoid.h"
#include "text/to_string.hpp"

namespace ncv
{
        struct function_rotated_ellipsoid_t : public function_t
        {
                explicit function_rotated_ellipsoid_t(const size_t dims)
                        :       m_dims(dims)
                {
                }

                virtual string_t name() const override
                {
                        return "rotated ellipsoid" + text::to_string(m_dims) + "D";
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
                                        for (size_t j = 0; j <= i; j ++)
                                        {
                                                fx += x(j) * x(j);
                                        }
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx.resize(m_dims);
                                gx.setZero();
                                for (size_t i = 0; i < m_dims; i ++)
                                {
                                        for (size_t j = 0; j <= i; j ++)
                                        {
                                                gx(j) += 2.0 * x(j);
                                        }
                                }

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return x.lpNorm<Eigen::Infinity>() < 65.536;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        return (x - vector_t::Zero(m_dims)).lpNorm<Eigen::Infinity>() < epsilon;
                }

                size_t  m_dims;
        };

        functions_t make_rotated_ellipsoid_funcs(size_t max_dims)
        {
                functions_t functions;

                for (size_t dims = 2; dims <= max_dims; dims *= 2)
                {
                        functions.push_back(std::make_shared<function_rotated_ellipsoid_t>(dims));
                }

                return functions;
        }
}
	
