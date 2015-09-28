#include "function_goldstein_price.h"

namespace ncv
{
        struct function_goldstein_price_t : public function_t
        {
                virtual string_t name() const override
                {
                        return "Goldstein-Price";
                }

                virtual opt_problem_t problem() const override
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return 2;
                        };

                        const opt_opfval_t fn_fval = [=] (const vector_t& x)
                        {
                                const long double a = x(0), b = x(1);

                                const long double z0 = 1 + a + b;
                                const long double z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
                                const long double z2 = 2 * a - 3 * b;
                                const long double z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

                                const long double u = 1 + z0 * z0 * z1;
                                const long double v = 30 + z2 * z2 * z3;

                                return static_cast<scalar_t>(u * v);
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                const long double a = x(0), b = x(1);

                                const long double z0 = 1 + a + b;
                                const long double z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
                                const long double z2 = 2 * a - 3 * b;
                                const long double z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

                                const long double u = 1 + z0 * z0 * z1;
                                const long double v = 30 + z2 * z2 * z3;

                                const long double z0da = 1;
                                const long double z0db = 1;

                                const long double z1da = -14 + 6 * a + 6 * b;
                                const long double z1db = -14 + 6 * a + 6 * b;

                                const long double z2da = +2;
                                const long double z2db = -3;

                                const long double z3da = -32 + 24 * a - 36 * b;
                                const long double z3db = +48 - 36 * a + 54 * b;

                                gx.resize(2);
                                gx(0) = static_cast<scalar_t>(
                                        u * z2 * (2 * z2da * z3 + z2 * z3da) +
                                        v * z0 * (2 * z0da * z1 + z0 * z1da));
                                gx(1) = static_cast<scalar_t>(
                                        u * z2 * (2 * z2db * z3 + z2 * z3db) +
                                        v * z0 * (2 * z0db * z1 + z0 * z1db));

                                return static_cast<scalar_t>(u * v);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return norm(x) < 2.0;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        const auto xmins =
                        {
                                scalars_t{ +0.0, -1.0 },
                                scalars_t{ +1.2, +0.8 },
                                scalars_t{ +1.8, +0.2 },
                                scalars_t{ -0.6, -0.4 }
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

        functions_t make_goldstein_price_funcs()
        {
                return { std::make_shared<function_goldstein_price_t>() };
        }
}
	
