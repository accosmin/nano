#include "util.h"
#include "sargan.h"

namespace nano
{
        function_sargan_t::function_sargan_t(const tensor_size_t dims) :
                m_dims(dims)
        {
        }

        std::string function_sargan_t::name() const
        {
                return "Sargan" + std::to_string(m_dims) + "D";
        }

        problem_t function_sargan_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        return  scalar_t(0.6) * x.array().square().sum() +
                                scalar_t(0.4) * nano::square(x.array().sum());
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx = scalar_t(1.2) * x.array() + scalar_t(0.8) * x.array().sum();

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_sargan_t::is_valid(const vector_t&) const
        {
                return true;
        }

        bool function_sargan_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
        }

        bool function_sargan_t::is_convex() const
        {
                return true;
        }

        tensor_size_t function_sargan_t::min_dims() const
        {
                return 1;
        }

        tensor_size_t function_sargan_t::max_dims() const
        {
                return 100 * 1000;
        }
}
