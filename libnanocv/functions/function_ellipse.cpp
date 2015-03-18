#include "function_ellipse.h"
#include "../util/random.hpp"

namespace ncv
{
        std::vector<function_t> make_ellipse_funcs(ncv::size_t max_dims)
        {
                std::vector<function_t> functions;

                for (size_t dims = 1; dims <= max_dims; dims *= 2)
                {
                        vector_t weights(dims);

                        random_t<scalar_t> rng(1.0, 1e+6);
                        rng(weights.data(), weights.data() + weights.size());

                        const opt_opsize_t fn_size = [=] ()
                        {
                                return dims;
                        };

                        const opt_opfval_t fn_fval = [=] (const vector_t& x)
                        {
                                scalar_t fx = 0;
                                for (size_t i = 0; i < dims; i ++)
                                {
                                        fx += x(i) * x(i) * weights(i);
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx.resize(dims);
                                for (size_t i = 0; i < dims; i ++)
                                {
                                        gx(i) = 2.0 * x(i) * weights(i);
                                }

                                return fn_fval(x);
                        };

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                solutions.emplace_back(vector_t::Zero(dims), 0);
                        }

                        functions.emplace_back("ellipse" + text::to_string(dims) + "D",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
