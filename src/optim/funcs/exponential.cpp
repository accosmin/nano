#include "util.hpp"
#include "exponential.h"

namespace nano
{
        function_exponential_t::function_exponential_t(const tensor_size_t dims) :
                m_dims(dims)
        {
        }

        std::string function_exponential_t::name() const
        {
                return "Exponential" + std::to_string(m_dims) + "D";
        }

        problem_t function_exponential_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        return -std::exp(-scalar_t(0.5) * x.array().square().sum());
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const auto fx = fn_fval(x);

                        gx = -fx * x;

                        return fx;
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_exponential_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < scalar_t(1);
        }

        bool function_exponential_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
        }
}
