#include "function_3hump_camel.h"

namespace ncv
{
        struct function_3hump_camel_t : public function_t
        {
                virtual string_t name() const override
                {
                        return "3hump camel";
                }

                virtual opt_problem_t problem() const override
                {
                        const opt_opsize_t fn_size = [] ()
                        {
                                return 2;
                        };

                        const opt_opfval_t fn_fval = [] (const vector_t& x)
                        {
                                const scalar_t a = x(0), b = x(1);

                                const scalar_t a2 = a * a;
                                const scalar_t a4 = a2 * a2;
                                const scalar_t a6 = a4 * a2;

                                return 2 * a2 - 1.05 * a4 + a6 / 6.0 + a * b + b * b;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                const scalar_t a = x(0), b = x(1);

                                const scalar_t a2 = a * a;
                                const scalar_t a3 = a * a2;
                                const scalar_t a5 = a3 * a2;

                                gx.resize(2);
                                gx(0) = 4 * a - 1.05 * 4 * a3 + a5 + b;
                                gx(1) = a + 2 * b;

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return  -5.0 <= x(0) && x(0) <= 5.0 &&
                                -5.0 <= x(1) && x(1) <= 5.0;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        const vector_t xmin = vector_t::Zero(2);
//                        const scalar_t fmin = 0.0;

                        return (x - xmin).lpNorm<Eigen::Infinity>() < epsilon;
                }
        };

        functions_t make_3hump_camel_funcs()
        {
                return { std::make_shared<function_3hump_camel_t>() };
        }
}
	
