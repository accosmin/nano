#include "function_rotated_ellipsoid.h"
#include "libnanocv/text/to_string.hpp"

namespace ncv
{
        std::vector<function_t> make_rotated_ellipsoid_funcs(ncv::size_t max_dims)
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
                                        for (size_t j = 0; j <= i; j ++)
                                        {
                                                fx += x(j) * x(j);
                                        }
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx.resize(dims);
                                gx.setZero();
                                for (size_t i = 0; i < dims; i ++)
                                {
                                        for (size_t j = 0; j <= i; j ++)
                                        {
                                                gx(j) += 2.0 * x(j);
                                        }
                                }

                                return fn_fval(x);
                        };

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                solutions.emplace_back(vector_t::Zero(dims), 0);
                        }

                        functions.emplace_back("rotated ellipsoid" + text::to_string(dims) + "D",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
