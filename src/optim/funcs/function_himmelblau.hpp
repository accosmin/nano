#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create Himmelblau test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        struct function_himmelblau_t : public function_t
        {
                virtual std::string name() const override
                {
                        return "Himmelblau";
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

                                const auto u = a * a + b - 11;
                                const auto v = a + b * b - 7;

                                return u * u + v * v;
                        };

                        const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                const auto a = x(0), b = x(1);

                                const auto u = a * a + b - 11;
                                const auto v = a + b * b - 7;

                                gx.resize(2);
                                gx(0) = 2 * u * 2 * a + 2 * v;
                                gx(1) = 2 * u + 2 * v * 2 * b;

                                return fn_fval(x);
                        };

                        return {fn_size, fn_fval, fn_grad};
                }

                virtual bool is_valid(const vector_t&) const override
                {
                        return true;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        const auto xmins =
                        {
                                std::vector<scalar_t>{ scalar_t(3.0), scalar_t(2.0) },
                                std::vector<scalar_t>{ scalar_t(-2.805118), scalar_t(3.131312) },
                                std::vector<scalar_t>{ scalar_t(-3.779310), scalar_t(-3.283186) },
                                std::vector<scalar_t>{ scalar_t(3.584428), scalar_t(-1.848126) }
                        };

                        return util::check_close(x, xmins, epsilon);
                }
        };
}
