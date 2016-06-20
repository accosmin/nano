#include "zakharov.h"
#include "util.hpp"

namespace nano
{
        function_zakharov_t::function_zakharov_t(const tensor_size_t dims) :
                m_dims(dims),
                m_weights(dims)
        {
                for (tensor_size_t i = 0; i < dims; i ++)
                {
                        m_weights(i) = scalar_t(i) / scalar_t(2) / scalar_t(dims);
                }
        }

        std::string function_zakharov_t::name() const
        {
                return "Zakharov" + std::to_string(m_dims) + "D";
        }

        problem_t function_zakharov_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        const scalar_t u = x.array().square().sum();
                        const scalar_t v = (m_weights.array() * x.array()).sum();

                        return u + nano::square(v) + nano::quartic(v);
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const scalar_t u = x.array().square().sum();
                        const scalar_t v = (m_weights.array() * x.array()).sum();

                        gx = 2 * x + (2 * v + 4 * nano::cube(v)) * m_weights;

                        return u + nano::square(v) + nano::quartic(v);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_zakharov_t::is_valid(const vector_t& x) const
        {
                return scalar_t(-5.0) < x.minCoeff() && x.maxCoeff() < scalar_t(10.0);
        }

        bool function_zakharov_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
        }
}
