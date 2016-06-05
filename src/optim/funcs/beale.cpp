#include "beale.h"
#include "util.hpp"

namespace nano
{
        std::string function_beale_t::name() const
        {
                return "Beale";
        }

        problem_t function_beale_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return vector_t::Index(2);
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        const auto a = x(0);
                        const auto b = x(1), b2 = b * b, b3 = b2 * b;

                        const auto z0 = scalar_t(1.5) - a + a * b;
                        const auto z1 = scalar_t(2.25) - a + a * b2;
                        const auto z2 = scalar_t(2.625) - a + a * b3;

                        return z0 * z0 + z1 * z1 + z2 * z2;
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const auto a = x(0);
                        const auto b = x(1), b2 = b * b, b3 = b2 * b;

                        const auto z0 = scalar_t(1.5) - a + a * b;
                        const auto z1 = scalar_t(2.25) - a + a * b2;
                        const auto z2 = scalar_t(2.625) - a + a * b3;

                        gx.resize(2);
                        gx(0) = 2 * (z0 * (-1 + b) + z1 * (-1 + b2) + z2 * (-1 + b3));
                        gx(1) = 2 * (z0 * (a) + z1 * (2 * a * b) + z2 * (3 * a * b2));

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_beale_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < scalar_t(4.5);
        }

        bool function_beale_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                const auto xmins =
                {
                        std::vector<scalar_t>{ scalar_t(3.0), scalar_t(0.5) }
                };

                return util::check_close(x, xmins, epsilon);
        }
}
