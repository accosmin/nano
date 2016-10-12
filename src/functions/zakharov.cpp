#include "util.h"
#include "zakharov.h"

namespace nano
{
        function_zakharov_t::function_zakharov_t(const tensor_size_t dims) :
                m_dims(dims)
        {
        }

        std::string function_zakharov_t::name() const
        {
                return "Zakharov" + std::to_string(m_dims) + "D";
        }

        problem_t function_zakharov_t::problem() const
        {
                const vector_t bias = vector_t::LinSpaced(m_dims, scalar_t(1) / scalar_t(2), scalar_t(m_dims) / scalar_t(2));

                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        const scalar_t u = x.array().square().sum();
                        const scalar_t v = (bias.array() * x.array()).sum();

                        return u + nano::square(v) + nano::quartic(v);
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const scalar_t u = x.array().square().sum();
                        const scalar_t v = (bias.array() * x.array()).sum();

                        gx = 2 * x.array() + (2 * v + 4 * nano::cube(v)) * bias.array();

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

        bool function_zakharov_t::is_convex() const
        {
                return true;
        }

        tensor_size_t function_zakharov_t::min_dims() const
        {
                return 2;
        }

        tensor_size_t function_zakharov_t::max_dims() const
        {
                return 100 * 1000;
        }
}
