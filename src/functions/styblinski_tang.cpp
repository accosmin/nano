#include "util.h"
#include "styblinski_tang.h"

namespace nano
{
        function_styblinski_tang_t::function_styblinski_tang_t(const tensor_size_t dims) :
                m_dims(dims)
        {
        }

        std::string function_styblinski_tang_t::name() const
        {
                return "Styblinski-Tang" + std::to_string(m_dims) + "D";
        }

        problem_t function_styblinski_tang_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        return (x.array().square().square() - 16 * x.array().square() + 5 * x.array()).sum();
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx = 4 * x.array().cube() - 32 * x.array() + 5;

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_styblinski_tang_t::is_valid(const vector_t& x) const
        {
                return scalar_t(-5.0) < x.minCoeff() && x.maxCoeff() < scalar_t(5.0);
        }

        bool function_styblinski_tang_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                const auto u1 = scalar_t(-2.9035340);
                const auto u2 = scalar_t(+2.7468027);

                bool ok = true;
                for (tensor_size_t i = 0; i < m_dims && ok; i ++)
                {
                        ok = std::fabs(x(i) - u1) < epsilon || std::fabs(x(i) - u2) < epsilon;
                }

                return ok;
        }

        bool function_styblinski_tang_t::is_convex() const
        {
                return false;
        }

        tensor_size_t function_styblinski_tang_t::min_dims() const
        {
                return 1;
        }

        tensor_size_t function_styblinski_tang_t::max_dims() const
        {
                return 100 * 1000;
        }
}
