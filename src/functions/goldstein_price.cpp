#include "util.h"
#include "goldstein_price.h"

namespace nano
{
        std::string function_goldstein_price_t::name() const
        {
                return "Goldstein-Price";
        }

        problem_t function_goldstein_price_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return 2;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        const auto a = x(0), b = x(1);

                        const auto z0 = 1 + a + b;
                        const auto z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
                        const auto z2 = 2 * a - 3 * b;
                        const auto z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

                        const auto u = 1 + z0 * z0 * z1;
                        const auto v = 30 + z2 * z2 * z3;

                        return u * v;
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const auto a = x(0), b = x(1);

                        const auto z0 = 1 + a + b;
                        const auto z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
                        const auto z2 = 2 * a - 3 * b;
                        const auto z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

                        const auto u = 1 + z0 * z0 * z1;
                        const auto v = 30 + z2 * z2 * z3;

                        const auto z0da = 1;
                        const auto z0db = 1;

                        const auto z1da = -14 + 6 * a + 6 * b;
                        const auto z1db = -14 + 6 * a + 6 * b;

                        const auto z2da = +2;
                        const auto z2db = -3;

                        const auto z3da = -32 + 24 * a - 36 * b;
                        const auto z3db = +48 - 36 * a + 54 * b;

                        gx.resize(2);
                        gx(0) = u * z2 * (2 * z2da * z3 + z2 * z3da) + v * z0 * (2 * z0da * z1 + z0 * z1da);
                        gx(1) = u * z2 * (2 * z2db * z3 + z2 * z3db) + v * z0 * (2 * z0db * z1 + z0 * z1db);

                        return u * v;
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_goldstein_price_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < 2.0;
        }

        bool function_goldstein_price_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                const auto xmins =
                {
                        std::vector<scalar_t>{ scalar_t(+0.0), scalar_t(-1.0) },
                        std::vector<scalar_t>{ scalar_t(+1.2), scalar_t(+0.8) },
                        std::vector<scalar_t>{ scalar_t(+1.8), scalar_t(+0.2) },
                        std::vector<scalar_t>{ scalar_t(-0.6), scalar_t(-0.4) }
                };

                return util::check_close(x, xmins, epsilon);
        }

        bool function_goldstein_price_t::is_convex() const
        {
                return false;
        }

        tensor_size_t function_goldstein_price_t::min_dims() const
        {
                return 2;
        }

        tensor_size_t function_goldstein_price_t::max_dims() const
        {
                return 2;
        }
}
