#include "function_mccormick.h"
#include <cmath>

namespace ncv
{
        struct function_mccormick_t : public function_t
        {
                virtual string_t name() const override
                {
                        return "McCormick";
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

                                return sin(a + b) + (a - b) * (a - b) - 1.5 * a + 2.5 * b + 1;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                const scalar_t a = x(0), b = x(1);

                                gx.resize(2);
                                gx(0) = cos(a + b) + 2 * (a - b) - 1.5;
                                gx(1) = cos(a + b) - 2 * (a - b) + 2.5;

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return  -1.5 < x(0) && x(0) < 4.0 &&
                                -3.0 < x(1) && x(1) < 4.0;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        const auto xmins =
                        {
                                scalars_t{ -0.54719, -1.54719 }
                        };

                        for (const auto& xmin : xmins)
                        {
                                if ((tensor::map_vector(xmin.data(), 2) - x).lpNorm<Eigen::Infinity>() < epsilon)
                                {
                                        return true;
                                }
                        }

                        return false;
                }
        };


        functions_t make_mccormick_funcs()
        {
                return { std::make_shared<function_mccormick_t>() };
        }
}
	
