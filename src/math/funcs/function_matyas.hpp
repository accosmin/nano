#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create Matyas test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        struct function_matyas_t : public function_t
        {
                virtual std::string name() const override
                {
                        return "Matyas";
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

                virtual bool is_valid(const vector_t& x) const override
                {
                        return util::norm(x) < scalar_t(10);
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        return util::distance(x, vector_t::Zero(2)) < epsilon;
                }
        };
}
