#include "test_func_goldstein_price.h"

namespace test
{
        using namespace ncv;

        std::vector<function_t> make_goldstein_price_funcs()
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

                                const scalar_t z0 = 1.0 + a + b;
                                const scalar_t z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
                                const scalar_t z2 = 2 * a - 3 * b;
                                const scalar_t z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

                                return (1 + z0 * z0 * z1) * (30 + z2 * z2 * z3);
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                const scalar_t a = x(0), b = x(1);

                                const scalar_t z0 = 1.0 + a + b;
                                const scalar_t z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
                                const scalar_t z2 = 2 * a - 3 * b;
                                const scalar_t z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

                                const scalar_t u = 1 + z0 * z0 * z1;
                                const scalar_t v = 30 + z2 * z2 * z3;

                                const scalar_t z0da = 1;
                                const scalar_t z0db = 1;

                                const scalar_t z1da = -14 + 6 * a + 6 * b;
                                const scalar_t z1db = -14 + 6 * a + 6 * b;

                                const scalar_t z2da = 2;
                                const scalar_t z2db = -3;

                                const scalar_t z3da = -32 + 24 * a - 36 * b;
                                const scalar_t z3db = 48 - 36 * a + 54 * b;

                                gx.resize(2);
                                gx(0) = u * (2 * z2 * z2da * z3 + z2 * z2 * z3da) +
                                        v * (2 * z0 * z0da * z1 + z0 * z0 * z1da);
                                gx(1) = u * (2 * z2 * z2db * z3 + z2 * z2 * z3db) +
                                        v * (2 * z0 * z0db * z1 + z0 * z0 * z1db);

                                return fn_fval(x);
                        };

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                vector_t x(2);
                                x(0) = +0.0;
                                x(1) = -1.0;
                                solutions.emplace_back(x, 3.0);
                        }
                        {
                                vector_t x(2);
                                x(0) = +1.2;
                                x(1) = +0.8;
                                solutions.emplace_back(x, 840.0);
                        }
                        {
                                vector_t x(2);
                                x(0) = +1.8;
                                x(1) = +0.2;
                                solutions.emplace_back(x, 84.0);
                        }
                        {
                                vector_t x(2);
                                x(0) = -0.6;
                                x(1) = -0.4;
                                solutions.emplace_back(x, 30.0);
                        }

                        functions.emplace_back("Goldstein-Price",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
