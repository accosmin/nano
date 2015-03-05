#include "function_rosenbrock.h"
#include "libnanocv/util/math.hpp"

namespace ncv
{
        std::vector<function_t> make_rosenbrock_funcs()
        {
                std::vector<function_t> functions;

                for (size_t dims = 2; dims <= 3; dims ++)
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return dims;
                        };

                        const opt_opfval_t fn_fval = [=] (const vector_t& x)
                        {
                                scalar_t fx = 0;
                                for (size_t i = 0; i + 1 < dims; i ++)
                                {
                                        fx += 100.0 * math::square(x(i + 1) - x(i) * x(i)) + math::square(x(i) - 1);
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx.resize(dims);
                                gx.setZero();
                                for (size_t i = 0; i + 1 < dims; i ++)
                                {
                                        gx(i) += 2.0 * (x(i) - 1);
                                        gx(i) += 100.0 * 2.0 * (x(i + 1) - x(i) * x(i)) * (- 2.0 * x(i));
                                        gx(i + 1) += 100.0 * 2.0 * (x(i + 1) - x(i) * x(i));
                                }

                                return fn_fval(x);
                        };

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                solutions.emplace_back(vector_t::Ones(dims), 0);
                                if (dims >= 4 && dims <= 7)
                                {
                                        vector_t x = vector_t::Ones(dims);
                                        x(0) = -1;
                                        solutions.emplace_back(x, 0);
                                }
                        }

                        functions.emplace_back("Rosenbrock" + text::to_string(dims) + "D",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
