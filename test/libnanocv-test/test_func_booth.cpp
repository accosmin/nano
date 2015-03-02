#include "test_func_booth.h"

namespace test
{
        using namespace ncv;

        std::vector<function_t> make_booth_funcs()
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

                                const scalar_t u = a + 2 * b - 7;
                                const scalar_t v = 2 * a + b - 5;

                                return u * u + v * v;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                const scalar_t a = x(0), b = x(1);

                                const scalar_t u = a + 2 * b - 7;
                                const scalar_t v = 2 * a + b - 5;

                                gx.resize(2);
                                gx(0) = 2 * u + 2 * v * 2;
                                gx(1) = 2 * u * 2 + 2 * v;

                                return fn_fval(x);
                        };

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                vector_t x(2);
                                x(0) = 1.0;
                                x(1) = 3.0;
                                solutions.emplace_back(x, 0.0);
                        }

                        functions.emplace_back("Booth",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
