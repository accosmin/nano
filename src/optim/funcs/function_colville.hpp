#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create Colville test functions
        ///
        struct function_colville_t : public function_t
        {
                explicit function_colville_t()
                {
                }

                virtual std::string name() const override
                {
                        return "Colville";
                }

                virtual problem_t problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return vector_t::Index(4);
                        };

                        const auto fn_fval = [=] (const vector_t& x)
                        {
                                const auto x1 = x(0);
                                const auto x2 = x(1);
                                const auto x3 = x(2);
                                const auto x4 = x(3);

                                return  100 * nano::square(x1 * x1 - x2) +
                                        nano::square(x1 - 1) +
                                        nano::square(x3 - 1) +
                                        90 * nano::square(x3 * x3 - x4) +
                                        scalar_t(10.1) * nano::square(x2 - 1) +
                                        scalar_t(10.1) * nano::square(x4 - 1) +
                                        scalar_t(19.8) * (x2 - 1) * (x4 - 1);
                        };

                        const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                const auto x1 = x(0);
                                const auto x2 = x(1);
                                const auto x3 = x(2);
                                const auto x4 = x(3);

                                gx.resize(4);
                                gx(0) = 400 * (x1 * x1 - x2) * x1 + 2 * (x1 - 1);
                                gx(1) = -200 * (x1 * x1 - x2) + scalar_t(20.2) * (x2 - 1) + scalar_t(19.8) * (x4 - 1);
                                gx(2) = 360 * (x3 * x3 - x4) * x3 + 2 * (x3 - 1);
                                gx(3) = -180 * (x3 * x3 - x4) + scalar_t(20.2) * (x4 - 1) + scalar_t(19.8) * (x2 - 1);

                                return fn_fval(x);
                        };

                        return {fn_size, fn_fval, fn_grad};
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return -10.0 < x.minCoeff() && x.maxCoeff() < scalar_t(10);
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        return util::distance(x, vector_t::Ones(4)) < epsilon;
                }
        };
}
