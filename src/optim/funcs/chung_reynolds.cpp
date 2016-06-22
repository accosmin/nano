#include "chung_reynolds.h"
#include "util.hpp"

namespace nano
{
        function_chung_reynolds_t::function_chung_reynolds_t(const tensor_size_t dims) :
                m_dims(dims)
        {
        }

        std::string function_chung_reynolds_t::name() const
        {
                return "Chung-Reynolds" + std::to_string(m_dims) + "D";
        }

        problem_t function_chung_reynolds_t::problem() const
        {
                const auto scale = scalar_t(1) / scalar_t(m_dims);

                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        const auto u = scale * x.array().square().sum();
                        return u * u;
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const auto u = scale * x.array().square().sum();

                        gx = (4 * scale * u) * x;

                        return u * u;
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_chung_reynolds_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < scalar_t(1);
        }

        bool function_chung_reynolds_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
        }
}
