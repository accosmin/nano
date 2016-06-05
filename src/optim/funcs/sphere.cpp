#include "sphere.h"
#include "util.hpp"

namespace nano
{
        function_sphere_t::function_sphere_t(const tensor_size_t dims) :
                m_dims(dims)
        {
        }

        std::string function_sphere_t::name() const
        {
                return "sphere" + std::to_string(m_dims) + "D";
        }

        problem_t function_sphere_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        return x.array().square().sum();
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx = 2 * x;

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_sphere_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < 5.12;
        }

        bool function_sphere_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
        }
}
