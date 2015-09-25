#include "function_rosenbrock.h"
#include "math/numeric.hpp"
#include "text/to_string.hpp"

namespace ncv
{
        struct function_rosenbrock_t : public function_t
        {
                explicit function_rosenbrock_t(const size_t dims)
                        :       m_dims(dims)
                {
                }

                virtual string_t name() const override
                {
                        return "Rosenbrock" + text::to_string(m_dims) + "D";
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
                                for (size_t i = 0; i + 1 < m_dims; i ++)
                                {
                                        fx += 100.0 * math::square(x(i + 1) - x(i) * x(i)) + math::square(x(i) - 1);
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx.resize(m_dims);
                                gx.setZero();
                                for (size_t i = 0; i + 1 < m_dims; i ++)
                                {
                                        gx(i) += 2.0 * (x(i) - 1);
                                        gx(i) += 100.0 * 2.0 * (x(i + 1) - x(i) * x(i)) * (- 2.0 * x(i));
                                        gx(i + 1) += 100.0 * 2.0 * (x(i + 1) - x(i) * x(i));
                                }

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const vector_t&) const override
                {
                        return true;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const
                {
                        vector_t xmin = vector_t::Ones(m_dims);
                        if (m_dims >= 4 && m_dims <= 7)
                        {
                                xmin(0) = -1;
                        }

                        return (x - xmin).lpNorm<Eigen::Infinity>() < epsilon;
                }

                size_t  m_dims;
        };

        functions_t make_rosenbrock_funcs(size_t max_dims)
        {
                functions_t functions;

                for (size_t dims = 2; dims <= max_dims; dims ++)
                {
                        functions.push_back(std::make_shared<function_rosenbrock_t>(dims));
                }

                return functions;
        }
}
	
