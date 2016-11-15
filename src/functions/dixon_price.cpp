#include "util.h"
#include "dixon_price.h"

namespace nano
{
        function_dixon_price_t::function_dixon_price_t(const tensor_size_t dims) :
                m_dims(dims)
        {
        }

        std::string function_dixon_price_t::name() const
        {
                return "Dixon-Price" + std::to_string(m_dims) + "D";
        }

        problem_t function_dixon_price_t::problem() const
        {
                const vector_t bias = vector_t::LinSpaced(m_dims, scalar_t(1), scalar_t(m_dims));

                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        const auto xsegm0 = x.segment(0, m_dims - 1);
                        const auto xsegm1 = x.segment(1, m_dims - 1);

                        return  nano::square(x(0) - 1) +
                                (bias.segment(1, m_dims - 1).array() *
                                (2 * xsegm1.array().square() - xsegm0.array()).square()).sum();
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const auto xsegm0 = x.segment(0, m_dims - 1);
                        const auto xsegm1 = x.segment(1, m_dims - 1);
                        const auto weight = bias.segment(1, m_dims - 1).array() *
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

        bool function_dixon_price_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < scalar_t(10);
        }

        bool function_dixon_price_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                const auto p = problem();

                return std::fabs(p.value(x)) < epsilon;

                // NB: there are quite a few local minima that are not easy to compute!
                return true;

//                        vector_t xmin(m_dims);
//                        for (tensor_size_t i = 0; i < m_dims; i ++)
//                        {
//                                xmin(i) = std::pow(2.0, -1.0 + std::pow(2.0, -i));
//                        }

//                        return distance(x, xmin) < epsilon;
        }

        bool function_dixon_price_t::is_convex() const
        {
                return false;
        }

        tensor_size_t function_dixon_price_t::min_dims() const
        {
                return 2;
        }

        tensor_size_t function_dixon_price_t::max_dims() const
        {
                return 100 * 1000;
        }
}
