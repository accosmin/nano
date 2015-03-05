#include "function_mccormick.h"
#include <cmath>

namespace test
{
        using namespace ncv;

        std::vector<function_t> make_mccormick_funcs()
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

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                vector_t x(2);
                                x(0) = -0.54719;
                                x(1) = -1.54719;
                                solutions.emplace_back(x, -1.913223);
                        }

                        functions.emplace_back("McCormick",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
