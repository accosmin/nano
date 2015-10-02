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
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return 2;
                        };

                        const opt_opfval_t fn_fval = [=] (const opt_vector_t& x)
                        {
                                const opt_scalar_t a = x(0), b = x(1);

                                const opt_scalar_t a2 = a * a;
                                const opt_scalar_t a4 = a2 * a2;
                                const opt_scalar_t a6 = a4 * a2;

                                return 2 * a2 - 1.05 * a4 + a6 / 6.0 + a * b + b * b;
                        };

                        const opt_opgrad_t fn_grad = [=] (const opt_vector_t& x, opt_vector_t& gx)
                        {
                                const opt_scalar_t a = x(0), b = x(1);

                                const opt_scalar_t a2 = a * a;
                                const opt_scalar_t a3 = a * a2;
                                const opt_scalar_t a5 = a3 * a2;

                                gx.resize(2);
                                gx(0) = 4 * a - 1.05 * 4 * a3 + a5 + b;
                                gx(1) = a + 2 * b;

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const opt_vector_t& x) const override
                {
                        return norm(x) < 5.0;
                }

                virtual bool is_minima(const opt_vector_t& x, const opt_scalar_t epsilon) const override
                {
                        const auto a = opt_scalar_t(4.2);
                        const auto b = std::sqrt(opt_scalar_t(3.64));

                        const auto xmp = std::sqrt(0.5 * (a + b));
                        const auto xmn = std::sqrt(0.5 * (a - b));

                        const auto xmins =
                        {
                                std::vector<opt_scalar_t>{ 0.0, 0.0 },
                                std::vector<opt_scalar_t>{ xmp, -0.5 * xmp },
                                std::vector<opt_scalar_t>{ xmn, -0.5 * xmn },
                                std::vector<opt_scalar_t>{ -xmp, 0.5 * xmp },
                                std::vector<opt_scalar_t>{ -xmn, 0.5 * xmn }
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

        functions_t make_3hump_camel_funcs()
        {
                return { std::make_shared<function_3hump_camel_t>() };
        }
}
	
