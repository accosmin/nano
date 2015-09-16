#include "function_3hump_camel.h"

namespace ncv
{
        std::vector<function_t> make_3hump_camel_funcs()
        {
                std::vector<function_t> functions;

                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return 2;
                        };

                        const opt_opfval_t fn_fval = [=] (const vector_t& x)
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

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                solutions.emplace_back(vector_t::Zero(2), 0.0);
                        }

                        functions.emplace_back("3hump camel",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
