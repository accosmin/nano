#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
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
                        :       m_dims(dims), m_weights(dims)
                {
                        for (tsize i = 0; i < m_dims; ++ i)
                        {
                               m_weights(i) = tscalar(i + 1);
                        }
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
                                const auto xsegm0 = x.segment(0, m_dims - 1);
                                const auto xsegm1 = x.segment(1, m_dims - 1);

                                return  nano::square(x(0) - 1) +
                                        (m_weights.segment(1, m_dims - 1).array() *
                                        (2 * xsegm1.array().square() - xsegm0.array()).square()).sum();
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const auto xsegm0 = x.segment(0, m_dims - 1);
                                const auto xsegm1 = x.segment(1, m_dims - 1);
                                const auto weight = m_weights.segment(1, m_dims - 1).array() *
                                        2 * (2 * xsegm1.array().square() - xsegm0.array());

                                gx.resize(m_dims);
                                gx.setZero();
                                gx(0) = 2 * (x(0) - 1);
                                gx.segment(1, m_dims - 1).array() += weight * 4 * xsegm1.array();
                                gx.segment(0, m_dims - 1).array() -= weight;

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
                tvector m_weights;
        };
}
