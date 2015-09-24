#include "function_cauchy.h"
#include "text/to_string.hpp"

namespace ncv
{
        template
        <


        functions_t make_cauchy_funcs(ncv::size_t max_dims)
        {
                functions_t functions;

                for (size_t dims = 1; dims <= max_dims; dims *= 2)
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return dims;
                        };

                        const opt_opfval_t fn_fval = [=] (const vector_t& x)
                        {
                                return (1.0 + x.array().square()).log().sum();
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx = (2.0 * x.array()) / (1.0 + x.array().square());

                                return fn_fval(x);
                        };

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                solutions.emplace_back(vector_t::Zero(dims), 0);
                        }

                        functions.emplace_back("cauchy" + text::to_string(dims) + "D",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
