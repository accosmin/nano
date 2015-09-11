#include "function_dixon_price.h"
#include "libmath/numeric.hpp"
#include "libnanocv/text/to_string.hpp"

namespace ncv
{
        std::vector<function_t> make_dixon_price_funcs(ncv::size_t max_dims)
        {
                std::vector<function_t> functions;

                for (size_t dims = 1; dims <= max_dims; dims *= 2)
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
                                        if (i == 0)
                                        {
                                                fx += math::square(x(0) - 1.0);
                                        }
                                        else
                                        {
                                                fx += (i + 1) * math::square(2.0 * math::square(x(i)) - x(i - 1));
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
                                        if (i == 0)
                                        {
                                                gx(0) += 2.0 * (x(0) - 1.0);
                                        }
                                        else
                                        {
                                                const scalar_t delta = (i + 1) * 2.0 * (2.0 * math::square(x(i)) - x(i - 1));

                                                gx(i) += delta * 4.0 * x(i);
                                                gx(i - 1) += - delta;
                                        }
                                }

                                return fn_fval(x);
                        };

                        std::vector<std::pair<vector_t, scalar_t>> solutions;
                        {
                                const scalar_t fx = 0.0;

                                vector_t x(dims);
                                for (size_t i = 0; i < dims; i ++)
                                {
                                        x(i) = std::pow(2.0, -1.0 + std::pow(2.0, -i));
                                }

                                solutions.emplace_back(x, fx);
                        }

                        functions.emplace_back("Dixon-Price" + text::to_string(dims) + "D",
                                               fn_size, fn_fval, fn_grad, solutions);
                }

                return functions;
        }
}
	
