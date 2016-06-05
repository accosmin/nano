#include "matyas.h"
#include "util.hpp"

namespace nano
{
        std::string function_matyas_t::name() const
        {
                return "Matyas";
        }

        problem_t function_matyas_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return 2;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        const auto a = x(0), b = x(1);

                        return scalar_t(0.26) * (a * a + b * b) - scalar_t(0.48) * a * b;
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const auto a = x(0), b = x(1);

                        gx.resize(2);
                        gx(0) = scalar_t(0.26) * 2 * a - scalar_t(0.48) * b;
                        gx(1) = scalar_t(0.26) * 2 * b - scalar_t(0.48) * a;

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_matyas_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < scalar_t(10);
        }

        bool function_matyas_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                return util::distance(x, vector_t::Zero(2)) < epsilon;
        }
}
