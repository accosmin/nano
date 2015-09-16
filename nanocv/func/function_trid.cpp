#include "function_trid.h"
#include "math/numeric.hpp"
#include "text/to_string.hpp"

namespace ncv
{
        std::vector<function_t> make_trid_funcs(ncv::size_t max_dims)
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
                                        fx += math::square(x(i) - 1.0);
                                }
                                for (size_t i = 1; i < dims; i ++)
                                {
                                        fx -= x(i) * x(i - 1);
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx.resize(dims);
                                gx.setZero();
                                for (size_t i = 0; i < dims; i ++)
                                {
                                        gx(i) += 2.0 * (x(i) - 1.0);
                                }
                                for (size_t i = 1; i < dims; i ++)
                                {
                                        gx(i) -= x(i - 1);
                                        gx(i - 1) -= x(i);
                                }

                                return fn_fval(x);
                        };

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                /// \todo
                        }

                        functions.emplace_back("Trid" + text::to_string(dims) + "D",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
