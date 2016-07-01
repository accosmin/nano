#include "test.h"

#include "funcs/trid.h"
#include "funcs/qing.h"
#include "funcs/beale.h"
#include "funcs/booth.h"
#include "funcs/cauchy.h"
#include "funcs/sphere.h"
#include "funcs/matyas.h"
#include "funcs/powell.h"
#include "funcs/sargan.h"
#include "funcs/colville.h"
#include "funcs/zakharov.h"
#include "funcs/mccormick.h"
#include "funcs/himmelblau.h"
#include "funcs/rosenbrock.h"
#include "funcs/exponential.h"
#include "funcs/3hump_camel.h"
#include "funcs/dixon_price.h"
#include "funcs/bohachevsky.h"
#include "funcs/axis_ellipsoid.h"
#include "funcs/chung_reynolds.h"
#include "funcs/goldstein_price.h"
#include "funcs/styblinski_tang.h"
#include "funcs/rotated_ellipsoid.h"
#include "funcs/schumer_steiglitz.h"

namespace nano
{
        static void append(const rfunction_t& func, const tensor_size_t dims, rfunctions_t& funcs)
        {
                if (func->min_dims() <= dims && dims <= func->max_dims())
                {
                        funcs.push_back(func);
                }
        }

        rfunctions_t make_functions(const tensor_size_t min_dims, const tensor_size_t max_dims)
        {
                assert(min_dims >= 1);
                assert(min_dims <= max_dims);

                rfunctions_t funcs;
                for (tensor_size_t dims = min_dims; dims <= max_dims; )
                {
                        append(std::make_shared<function_beale_t>(), dims, funcs);
                        append(std::make_shared<function_booth_t>(), dims, funcs);
                        append(std::make_shared<function_matyas_t>(), dims, funcs);
                        append(std::make_shared<function_colville_t>(), dims, funcs);
                        append(std::make_shared<function_mccormick_t>(), dims, funcs);
                        append(std::make_shared<function_3hump_camel_t>(), dims, funcs);
                        append(std::make_shared<function_goldstein_price_t>(), dims, funcs);
                        append(std::make_shared<function_himmelblau_t>(), dims, funcs);
                        append(std::make_shared<function_bohachevsky_t>(function_bohachevsky_t::btype::one), dims, funcs);
                        append(std::make_shared<function_bohachevsky_t>(function_bohachevsky_t::btype::two), dims, funcs);
                        append(std::make_shared<function_bohachevsky_t>(function_bohachevsky_t::btype::three), dims, funcs);

                        append(std::make_shared<function_trid_t>(dims), dims, funcs);
                        append(std::make_shared<function_qing_t>(dims), dims, funcs);
                        append(std::make_shared<function_cauchy_t>(dims), dims, funcs);
                        append(std::make_shared<function_sargan_t>(dims), dims, funcs);
                        if (dims % 4 == 0)
                        {
                                append(std::make_shared<function_powell_t>(dims), dims, funcs);
                        }
                        append(std::make_shared<function_zakharov_t>(dims), dims, funcs);
                        append(std::make_shared<function_rosenbrock_t>(dims), dims, funcs);
                        append(std::make_shared<function_exponential_t>(dims), dims, funcs);
                        append(std::make_shared<function_dixon_price_t>(dims), dims, funcs);
                        append(std::make_shared<function_chung_reynolds_t>(dims), dims, funcs);
                        append(std::make_shared<function_axis_ellipsoid_t>(dims), dims, funcs);
                        append(std::make_shared<function_styblinski_tang_t>(dims), dims, funcs);
                        append(std::make_shared<function_sphere_t>(dims), dims, funcs);
                        append(std::make_shared<function_schumer_steiglitz_t>(dims), dims, funcs);
                        append(std::make_shared<function_rotated_ellipsoid_t>(dims), dims, funcs);

                        if (dims <= 8)
                        {
                                ++ dims;
                        }
                        else
                        {
                                dims *= 2;
                        }
                }

                return funcs;
        }

        rfunctions_t make_convex_functions(const tensor_size_t min_dims, const tensor_size_t max_dims)
        {
                auto funcs = make_functions(min_dims, max_dims);

                funcs.erase(
                        std::remove_if(funcs.begin(), funcs.end(), [] (const auto& func) { return !func->is_convex(); }),
                        funcs.end());

                return funcs;
        }
}
