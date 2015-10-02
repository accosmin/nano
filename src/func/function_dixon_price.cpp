#include "function_dixon_price.h"
#include "math/numeric.hpp"
#include "text/to_string.hpp"

namespace ncv
{
        struct function_dixon_price_t : public function_t
        {
                explicit function_dixon_price_t(const opt_size_t dims)
                        :       m_dims(dims)
                {
                }

                virtual string_t name() const override
                {
                        return "Dixon-Price" + text::to_string(m_dims) + "D";
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
                                        if (i == 0)
                                        {
                                                fx += math::square(x(0) - 1.0);
                                        }
                                        else
                                        {
                                                fx += (i + 1) * math::square(2.0 * math::square(x(i)) - x(i - 1));
                                        }
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const opt_vector_t& x, opt_vector_t& gx)
                        {
                                gx = opt_vector_t::Zero(m_dims);
                                for (opt_size_t i = 0; i < m_dims; i ++)
                                {
                                        if (i == 0)
                                        {
                                                gx(0) += 2.0 * (x(0) - 1.0);
                                        }
                                        else
                                        {
                                                const opt_scalar_t delta = (i + 1) * 2.0 * (2.0 * math::square(x(i)) - x(i - 1));

                                                gx(i) += delta * 4.0 * x(i);
                                                gx(i - 1) += - delta;
                                        }
                                }

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const opt_vector_t& x) const override
                {
                        return norm(x) < 10.0;
                }

                virtual bool is_minima(const opt_vector_t&, const opt_scalar_t) const override
                {
                        // NB: there are quite a few local minima that are not easy to compute!
                        return true;

//                        opt_vector_t xmin(m_dims);
//                        for (opt_size_t i = 0; i < m_dims; i ++)
//                        {
//                                xmin(i) = std::pow(2.0, -1.0 + std::pow(2.0, -i));
//                        }

//                        return distance(x, xmin) < epsilon;
                }

                opt_size_t      m_dims;
        };

        functions_t make_dixon_price_funcs(opt_size_t max_dims)
        {
                functions_t functions;

                for (opt_size_t dims = 1; dims <= max_dims; dims *= 2)
                {
                        functions.push_back(std::make_shared<function_dixon_price_t>(dims));
                }

                return functions;
        }
}
	
