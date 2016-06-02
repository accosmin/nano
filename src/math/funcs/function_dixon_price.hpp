#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create Dixon-Price test functions
        ///
        struct function_dixon_price_t : public function_t
        {
                explicit function_dixon_price_t(const tsize dims) :
                        m_dims(dims),
                        m_weights(dims)
                {
                        for (tsize i = 0; i < m_dims; ++ i)
                        {
                               m_weights(i) = scalar_t(i + 1);
                        }
                }

                virtual std::string name() const override
                {
                        return "Dixon-Price" + std::to_string(m_dims) + "D";
                }

                virtual problem_t problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const vector_t& x)
                        {
                                const auto xsegm0 = x.segment(0, m_dims - 1);
                                const auto xsegm1 = x.segment(1, m_dims - 1);

                                return  nano::square(x(0) - 1) +
                                        (m_weights.segment(1, m_dims - 1).array() *
                                        (2 * xsegm1.array().square() - xsegm0.array()).square()).sum();
                        };

                        const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
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

                        return {fn_size, fn_fval, fn_grad};
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return util::norm(x) < scalar_t(10);
                }

                virtual bool is_minima(const vector_t&, const scalar_t) const override
                {
                        // NB: there are quite a few local minima that are not easy to compute!
                        return true;

//                        vector_t xmin(m_dims);
//                        for (tsize i = 0; i < m_dims; i ++)
//                        {
//                                xmin(i) = std::pow(2.0, -1.0 + std::pow(2.0, -i));
//                        }

//                        return distance(x, xmin) < epsilon;
                }

                tsize   m_dims;
                vector_t m_weights;
        };
}
