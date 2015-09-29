#include "function_rosenbrock.h"
#include "math/numeric.hpp"
#include "text/to_string.hpp"

namespace ncv
{
        struct function_rosenbrock_t : public function_t
        {
                explicit function_rosenbrock_t(const opt_size_t dims)
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

                        const opt_opfval_t fn_fval = [=] (const opt_vector_t& x)
                        {
                                opt_scalar_t fx = 0;
                                for (opt_size_t i = 0; i + 1 < m_dims; i ++)
                                {
                                        fx += 100.0 * math::square(x(i + 1) - x(i) * x(i)) + math::square(x(i) - 1);
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const opt_vector_t& x, opt_vector_t& gx)
                        {
                                gx.resize(m_dims);
                                gx.setZero();
                                for (opt_size_t i = 0; i + 1 < m_dims; i ++)
                                {
                                        gx(i) += 2.0 * (x(i) - 1);
                                        gx(i) += 100.0 * 2.0 * (x(i + 1) - x(i) * x(i)) * (- 2.0 * x(i));
                                        gx(i + 1) += 100.0 * 2.0 * (x(i + 1) - x(i) * x(i));
                                }

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const opt_vector_t&) const override
                {
                        return true;
                }

                virtual bool is_minima(const opt_vector_t& x, const opt_scalar_t epsilon) const override
                {
                        {
                                const opt_vector_t xmin = opt_vector_t::Ones(m_dims);

                                if (distance(x, xmin) < epsilon)
                                {
                                        return true;
                                }
                        }

                        if (m_dims >= 4 && m_dims <= 7)
                        {
                                opt_vector_t xmin = opt_vector_t::Ones(m_dims);
                                xmin(0) = -1;

                                if (distance(x, xmin) < epsilon)
                                {
                                        return true;
                                }
                        }

                        return false;
                }

                opt_size_t      m_dims;
        };

        functions_t make_rosenbrock_funcs(opt_size_t max_dims)
        {
                functions_t functions;

                for (opt_size_t dims = 2; dims <= max_dims; dims ++)
                {
                        functions.push_back(std::make_shared<function_rosenbrock_t>(dims));
                }

                return functions;
        }
}
	
