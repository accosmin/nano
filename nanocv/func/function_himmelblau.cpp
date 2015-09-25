#include "function_himmelblau.h"

namespace ncv
{
        struct function_himmelblau_t : public function_t
        {
                virtual string_t name() const override
                {
                        return "Himmelblau";
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

                                const scalar_t u = a * a + b - 11;
                                const scalar_t v = a + b * b - 7;

                                return u * u + v * v;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                const scalar_t a = x(0), b = x(1);

                                const scalar_t u = a * a + b - 11;
                                const scalar_t v = a + b * b - 7;

                                gx.resize(2);
                                gx(0) = 2 * u * 2 * a + 2 * v;
                                gx(1) = 2 * u + 2 * v * 2 * b;

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const vector_t&) const override
                {
                        return true;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        const auto xmins =
                        {
                                scalars_t{ 3.0, 2.0 },
                                scalars_t{ -2.805118, 3.131312 },
                                scalars_t{ -3.779310, -3.283186 },
                                scalars_t{ 3.584428, -1.848126 }
                        };

                        for (const auto& xmin : xmins)
                        {
                                if (distance(x, tensor::map_vector(xmin.data(), 2)) < epsilon)
                                {
                                        return true;
                                }
                        }

                        return false;
                }
        };

        functions_t make_himmelblau_funcs()
        {
                return { std::make_shared<function_himmelblau_t>() };
        }
}
	
