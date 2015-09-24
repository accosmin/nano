#include "function_himmelblau.h"

namespace ncv
{
        functions_t make_himmelblau_funcs()
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

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                vector_t x(2);
                                x(0) = -0.270845;
                                x(1) = -0.923039;
                                solutions.emplace_back(x, 181.617);
                        }
                        {
                                vector_t x(2);
                                x(0) = 3.0;
                                x(1) = 2.0;
                                solutions.emplace_back(x, 0);
                        }
                        {
                                vector_t x(2);
                                x(0) = -2.805118;
                                x(1) = 3.131312;
                                solutions.emplace_back(x, 0);
                        }
                        {
                                vector_t x(2);
                                x(0) = -3.779310;
                                x(1) = -3.283186;
                                solutions.emplace_back(x, 0);
                        }
                        {
                                vector_t x(2);
                                x(0) = 3.584428;
                                x(1) = -1.848126;
                                solutions.emplace_back(x, 0);
                        }

                        functions.emplace_back("Himmelblau",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
