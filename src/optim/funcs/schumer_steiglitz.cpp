#include "util.hpp"
#include "math/numeric.hpp"
#include "schumer_steiglitz.h"

namespace nano
{
        function_schumer_steiglitz_t::function_schumer_steiglitz_t(const tensor_size_t dims) :
                m_dims(dims)
        {
        }

        std::string function_schumer_steiglitz_t::name() const
        {
                return "Schumer-Steiglitz" + std::to_string(m_dims) + "D";
        }

        problem_t function_schumer_steiglitz_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        return x.array().square().square().sum();
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx = 4 * x.array().cube();

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_schumer_steiglitz_t::is_valid(const vector_t&) const
        {
                return true;
        }

        bool function_schumer_steiglitz_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
        }
}
