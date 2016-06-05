#include "sum_squares.h"
#include "util.hpp"

namespace nano
{
        function_sum_squares_t::function_sum_squares_t(const tensor_size_t dims) :
                m_dims(dims),
                m_weights(dims)
        {
                for (tensor_size_t i = 0; i < dims; i ++)
                {
                        m_weights(i) = static_cast<scalar_t>(i + 1);
                }
        }

        std::string function_sum_squares_t::name() const
        {
                return "sum squares" + std::to_string(m_dims) + "D";
        }

        problem_t function_sum_squares_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        return (m_weights.array() * x.array().square()).sum();
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx = 2 * m_weights.array() * x.array();

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_sum_squares_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < scalar_t(5.12);
        }

        bool function_sum_squares_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
        }
}
