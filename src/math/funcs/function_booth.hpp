#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create Booth test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        struct function_booth_t : public function_t
        {
                virtual std::string name() const override
                {
                        return "Booth";
                }

                virtual problem_t problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return vector_t::Index(2);
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
                                gx(0) = 2 * u + 2 * v * 2;
                                gx(1) = 2 * u * 2 + 2 * v;

                                return fn_fval(x);
                        };

                        return {fn_size, fn_fval, fn_grad};
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return util::norm(x) < 10.0;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        const auto xmins =
                        {
                                std::vector<scalar_t>{ 1.0, 3.0 }
                        };

                        return util::check_close(x, xmins, epsilon);
                }
        };
}
