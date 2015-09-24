#include "function_matyas.h"

namespace ncv
{
        functions_t make_matyas_funcs()
        {
                functions_t functions;

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

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                solutions.emplace_back(vector_t::Zero(2), 0.0);
                        }

                        functions.emplace_back("Matyas",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
