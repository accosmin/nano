#include "function_sum_squares.h"
#include "text/to_string.hpp"

namespace ncv
{
        std::vector<function_t> make_sum_squares_funcs(ncv::size_t max_dims)
        {
                std::vector<function_t> functions;

                for (size_t dims = 2; dims <= max_dims; dims *= 2)
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return dims;
                        };

                        const opt_opfval_t fn_fval = [=] (const vector_t& x)
                        {
                                scalar_t fx = 0;
                                for (size_t i = 0; i < dims; i ++)
                                {
                                        fx += (i + 1) * x(i) * x(i);
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx.resize(dims);
                                for (size_t i = 0; i < dims; i ++)
                                {
                                        gx(i) = 2.0 * (i + 1) * x(i);
                                }

                                return fn_fval(x);
                        };

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                solutions.emplace_back(vector_t::Zero(dims), 0);
                        }

                        functions.emplace_back("sum squares" + text::to_string(dims) + "D",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
