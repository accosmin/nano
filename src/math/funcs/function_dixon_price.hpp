#pragma once

#include "util.hpp"
#include "function.hpp"

namespace math
{
        ///
        /// \brief create Dixon-Price test functions
        ///
        template
        <
                typename tscalar
        >
        struct function_dixon_price_t : public function_t<tscalar>
        {
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;

                explicit function_dixon_price_t(const tsize dims)
                        :       m_dims(dims)
                {
                }

                virtual std::string name() const override
                {
                        return "Dixon-Price" + std::to_string(m_dims) + "D";
                }

                virtual tproblem problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                tscalar fx = 0;
                                for (tsize i = 0; i < m_dims; i ++)
                                {
                                        if (i == 0)
                                        {
                                                fx += math::square(x(0) - 1);
                                        }
                                        else
                                        {
                                                fx += tscalar(i + 1) * math::square(2 * math::square(x(i)) - x(i - 1));
                                        }
                                }

                                return fx;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                gx = tvector::Zero(m_dims);
                                for (tsize i = 0; i < m_dims; i ++)
                                {
                                        if (i == 0)
                                        {
                                                gx(0) += 2 * (x(0) - 1);
                                        }
                                        else
                                        {
                                                const auto delta = tscalar(i + 1) * 2 * (2 * math::square(x(i)) - x(i - 1));

                                                gx(i) += delta * 4 * x(i);
                                                gx(i - 1) += - delta;
                                        }
                                }

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return util::norm(x) < tscalar(10);
                }

                virtual bool is_minima(const tvector&, const tscalar) const override
                {
                        // NB: there are quite a few local minima that are not easy to compute!
                        return true;

//                        tvector xmin(m_dims);
//                        for (tsize i = 0; i < m_dims; i ++)
//                        {
//                                xmin(i) = std::pow(2.0, -1.0 + std::pow(2.0, -i));
//                        }

//                        return distance(x, xmin) < epsilon;
                }

                tsize   m_dims;
        };
}