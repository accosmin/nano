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

                        const opt_opfval_t fn_fval = [=] (const opt_vector_t& x)
                        {
                                const opt_scalar_t a = x(0), b = x(1);

                                const opt_scalar_t z0 = 1 + a + b;
                                const opt_scalar_t z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
                                const opt_scalar_t z2 = 2 * a - 3 * b;
                                const opt_scalar_t z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

                                const opt_scalar_t u = 1 + z0 * z0 * z1;
                                const opt_scalar_t v = 30 + z2 * z2 * z3;

                                return u * v;
                        };

                        const opt_opgrad_t fn_grad = [=] (const opt_vector_t& x, opt_vector_t& gx)
                        {
                                const opt_scalar_t a = x(0), b = x(1);

                                const opt_scalar_t z0 = 1 + a + b;
                                const opt_scalar_t z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
                                const opt_scalar_t z2 = 2 * a - 3 * b;
                                const opt_scalar_t z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

                                const opt_scalar_t u = 1 + z0 * z0 * z1;
                                const opt_scalar_t v = 30 + z2 * z2 * z3;

                                const opt_scalar_t z0da = 1;
                                const opt_scalar_t z0db = 1;

                                const opt_scalar_t z1da = -14 + 6 * a + 6 * b;
                                const opt_scalar_t z1db = -14 + 6 * a + 6 * b;

                                const opt_scalar_t z2da = +2;
                                const opt_scalar_t z2db = -3;

                                const opt_scalar_t z3da = -32 + 24 * a - 36 * b;
                                const opt_scalar_t z3db = +48 - 36 * a + 54 * b;

                                gx.resize(2);
                                gx(0) = u * z2 * (2 * z2da * z3 + z2 * z3da) +
                                        v * z0 * (2 * z0da * z1 + z0 * z1da);
                                gx(1) = u * z2 * (2 * z2db * z3 + z2 * z3db) +
                                        v * z0 * (2 * z0db * z1 + z0 * z1db);

                                return u * v;
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const opt_vector_t& x) const override
                {
                        return norm(x) < 2.0;
                }

                virtual bool is_minima(const opt_vector_t& x, const opt_scalar_t epsilon) const override
                {
                        const auto xmins =
                        {
                                std::vector<opt_scalar_t>{ +0.0, -1.0 },
                                std::vector<opt_scalar_t>{ +1.2, +0.8 },
                                std::vector<opt_scalar_t>{ +1.8, +0.2 },
                                std::vector<opt_scalar_t>{ -0.6, -0.4 }
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
	
