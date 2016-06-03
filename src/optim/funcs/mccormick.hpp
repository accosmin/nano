#pragma once

#include "util.hpp"
#include "function.hpp"
#include <cmath>

namespace nano
{
        ///
        /// \brief create McCormick test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        struct function_mccormick_t : public function_t
        {
                virtual std::string name() const override
                {
                        return "McCormick";
                }

                virtual problem_t problem() const override
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

                virtual bool is_valid(const vector_t& x) const override
                {
                        return  scalar_t(-1.5) < x(0) && x(0) < scalar_t(4.0) &&
                                scalar_t(-3.0) < x(1) && x(1) < scalar_t(4.0);
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        const auto xmins =
                        {
                                std::vector<scalar_t>{ scalar_t(-0.54719), scalar_t(-1.54719) }
                        };

                        return util::check_close(x, xmins, epsilon);
                }
        };
}
