#include "booth.h"
#include "util.hpp"

namespace nano
{
        std::string function_booth_t::name() const
        {
                return "Booth";
        }

        problem_t function_booth_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return 2;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        const auto a = x(0), b = x(1);

                        const auto u = a + 2 * b - 7;
                        const auto v = 2 * a + b - 5;

                        return u * u + v * v;
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const auto a = x(0), b = x(1);

                        const auto u = a + 2 * b - 7;
                        const auto v = 2 * a + b - 5;

                        gx.resize(2);
                        gx(0) = 2 * u + 4 * v;
                        gx(1) = 4 * u + 2 * v;

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_booth_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < 10.0;
        }

        bool function_booth_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                const auto xmins =
                {
                        std::vector<scalar_t>{ 1.0, 3.0 }
                };

                return util::check_close(x, xmins, epsilon);
        }

        bool function_booth_t::is_convex() const
        {
                return false;
        }

        tensor_size_t function_booth_t::min_dims() const
        {
                return 2;
        }

        tensor_size_t function_booth_t::max_dims() const
        {
                return 2;
        }
}

