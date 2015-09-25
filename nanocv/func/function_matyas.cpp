#include "function_matyas.h"

namespace ncv
{
        struct function_matyas_t : public function_t
        {
                virtual string_t name() const override
                {
                        return "Matyas";
                }

                virtual opt_problem_t problem() const override
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return 2;
                        };

                        const opt_opfval_t fn_fval = [=] (const vector_t& x)
                        {
                                const scalar_t a = x(0), b = x(1);

                                return 0.26 * (a * a + b * b) - 0.48 * a * b;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                const scalar_t a = x(0), b = x(1);

                                gx.resize(2);
                                gx(0) = 0.26 * 2 * a - 0.48 * b;
                                gx(1) = 0.26 * 2 * b - 0.48 * a;

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return norm(x) < 10.0;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        return distance(x, vector_t::Zero(2)) < epsilon;
                }
        };

        functions_t make_matyas_funcs()
        {
                return { std::make_shared<function_matyas_t>() };
        }
}
	
