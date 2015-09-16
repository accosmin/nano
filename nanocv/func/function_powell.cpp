#include "function_powell.h"
#include "math/numeric.hpp"
#include "text/to_string.hpp"

namespace ncv
{
        std::vector<function_t> make_powell_funcs(ncv::size_t max_dims)
        {
                std::vector<function_t> functions;

                for (size_t dims = 4; dims <= max_dims; dims *= 4)
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return dims;
                        };

                        const opt_opfval_t fn_fval = [=] (const vector_t& x)
                        {
                                scalar_t fx = 0;
                                for (size_t i = 0, i4 = 0; i < dims / 4; i ++, i4 += 4)
                                {
                                        fx += math::square(x(i4 + 0) + x(i4 + 1) * 10.0);
                                        fx += math::square(x(i4 + 2) - x(i4 + 3)) * 5.0;
                                        fx += math::quartic(x(i4 + 1) - x(i4 + 2) * 2.0);
                                        fx += math::quartic(x(i4 + 0) - x(i4 + 3)) * 10.0;
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx.resize(dims);
                                gx.setZero();
                                for (size_t i = 0, i4 = 0; i < dims / 4; i ++, i4 += 4)
                                {
                                        const scalar_t gfx1 = (x(i4 + 0) + x(i4 + 1) * 10.0) * 2.0;
                                        const scalar_t gfx2 = (x(i4 + 2) - x(i4 + 3)) * 5.0 * 2.0;
                                        const scalar_t gfx3 = math::cube(x(i4 + 1) - x(i4 + 2) * 2.0) * 4.0;
                                        const scalar_t gfx4 = math::cube(x(i4 + 0) - x(i4 + 3)) * 10.0 * 4.0;

                                        gx(i4 + 0) += gfx1 + gfx4;
                                        gx(i4 + 1) += gfx1 * 10.0 + gfx3;
                                        gx(i4 + 2) += gfx2 - 2.0 * gfx3;
                                        gx(i4 + 3) += - gfx2 - gfx4;
                                }

                                return fn_fval(x);
                        };

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                solutions.emplace_back(vector_t::Zero(dims), 0);
                        }

                        functions.emplace_back("Powell" + text::to_string(dims) + "D",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
