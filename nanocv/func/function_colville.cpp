#include "function_colville.h"
#include "math/numeric.hpp"

namespace ncv
{
        struct function_colville_t : public function_t
        {
                explicit function_colville_t()
                {
                }

                virtual string_t name() const override
                {
                        return "Colville";
                }

                virtual opt_problem_t problem() const override
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return 4;
                        };

                        const opt_opfval_t fn_fval = [=] (const opt_vector_t& x)
                        {
                                const auto x1 = x(0);
                                const auto x2 = x(1);
                                const auto x3 = x(2);
                                const auto x4 = x(3);

                                return  100 * math::square(x1 * x1 - x2) +
                                        math::square(x1 - 1) +
                                        math::square(x3 - 1) +
                                        90 * math::square(x3 * x3 - x4) +
                                        10.1 * math::square(x2 - 1) +
                                        10.1 * math::square(x4 - 1) +
                                        19.8 * (x2 - 1) * (x4 - 1);
                        };

                        const opt_opgrad_t fn_grad = [=] (const opt_vector_t& x, opt_vector_t& gx)
                        {
                                const auto x1 = x(0);
                                const auto x2 = x(1);
                                const auto x3 = x(2);
                                const auto x4 = x(3);

                                gx.resize(4);
                                gx(0) = 400 * (x1 * x1 - x2) * x1 + 2 * (x1 - 1);
                                gx(1) = -200 * (x1 * x1 - x2) + 20.2 * (x2 - 1) + 19.8 * (x4 - 1);
                                gx(2) = 360 * (x3 * x3 - x4) * x3 + 2 * (x3 - 1);
                                gx(3) = -180 * (x3 * x3 - x4) + 20.2 * (x4 - 1) + 19.8 * (x2 - 1);

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const opt_vector_t& x) const override
                {
                        return -10.0 < x.minCoeff() && x.maxCoeff() < 10.0;
                }

                virtual bool is_minima(const opt_vector_t& x, const opt_scalar_t epsilon) const override
                {
                        return distance(x, opt_vector_t::Constant(1.0, 2)) < epsilon;
                }
        };

        functions_t make_colville_funcs()
        {
                return { std::make_shared<function_colville_t>() };
        }
}
	
