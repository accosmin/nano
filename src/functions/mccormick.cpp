#include "mccormick.h"
#include "util.hpp"
#include <cmath>

namespace nano
{
        std::string function_mccormick_t::name() const
        {
                return "McCormick";
        }

        problem_t function_mccormick_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return 2;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        const auto a = x(0), b = x(1);

                        return std::sin(a + b) + (a - b) * (a - b) - scalar_t(1.5) * a + scalar_t(2.5) * b + 1;
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const auto a = x(0), b = x(1);

                        gx.resize(2);
                        gx(0) = std::cos(a + b) + 2 * (a - b) - scalar_t(1.5);
                        gx(1) = std::cos(a + b) - 2 * (a - b) + scalar_t(2.5);

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_mccormick_t::is_valid(const vector_t& x) const
        {
                return  scalar_t(-1.5) < x(0) && x(0) < scalar_t(4.0) &&
                        scalar_t(-3.0) < x(1) && x(1) < scalar_t(4.0);
        }

        bool function_mccormick_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                const auto xmins =
                {
                        std::vector<scalar_t>{ scalar_t(-0.54719), scalar_t(-1.54719) }
                };

                return util::check_close(x, xmins, epsilon);
        }

        bool function_mccormick_t::is_convex() const
        {
                return false;
        }

        tensor_size_t function_mccormick_t::min_dims() const
        {
                return 2;
        }

        tensor_size_t function_mccormick_t::max_dims() const
        {
                return 2;
        }
}
