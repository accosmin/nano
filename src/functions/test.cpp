#include "test.h"

#include "trid.h"
#include "qing.h"
#include "beale.h"
#include "booth.h"
#include "cauchy.h"
#include "sphere.h"
#include "matyas.h"
#include "powell.h"
#include "sargan.h"
#include "colville.h"
#include "zakharov.h"
#include "mccormick.h"
#include "himmelblau.h"
#include "rosenbrock.h"
#include "exponential.h"
#include "3hump_camel.h"
#include "dixon_price.h"
#include "bohachevsky.h"
#include "axis_ellipsoid.h"
#include "chung_reynolds.h"
#include "goldstein_price.h"
#include "styblinski_tang.h"
#include "rotated_ellipsoid.h"
#include "schumer_steiglitz.h"

namespace nano
{
        static void append(rfunction_t&& func, const tensor_size_t dims, rfunctions_t& funcs)
        {
                if (func->min_dims() <= dims && dims <= func->max_dims())
                {
                        funcs.push_back(std::move(func));
                }
        }

        rfunctions_t make_functions(const tensor_size_t min_dims, const tensor_size_t max_dims)
        {
                assert(min_dims >= 1);
                assert(min_dims <= max_dims);

                rfunctions_t funcs;
                for (tensor_size_t dims = min_dims; dims <= max_dims; )
                {
                        append(std::make_unique<function_beale_t>(), dims, funcs);
                        append(std::make_unique<function_booth_t>(), dims, funcs);
                        append(std::make_unique<function_matyas_t>(), dims, funcs);
                        append(std::make_unique<function_colville_t>(), dims, funcs);
                        append(std::make_unique<function_mccormick_t>(), dims, funcs);
                        append(std::make_unique<function_3hump_camel_t>(), dims, funcs);
                        append(std::make_unique<function_goldstein_price_t>(), dims, funcs);
                        append(std::make_unique<function_himmelblau_t>(), dims, funcs);
                        append(std::make_unique<function_bohachevsky_t>(function_bohachevsky_t::btype::one), dims, funcs);
                        append(std::make_unique<function_bohachevsky_t>(function_bohachevsky_t::btype::two), dims, funcs);
                        append(std::make_unique<function_bohachevsky_t>(function_bohachevsky_t::btype::three), dims, funcs);

                        append(std::make_unique<function_trid_t>(dims), dims, funcs);
                        append(std::make_unique<function_qing_t>(dims), dims, funcs);
                        append(std::make_unique<function_cauchy_t>(dims), dims, funcs);
                        append(std::make_unique<function_sargan_t>(dims), dims, funcs);
                        if (dims % 4 == 0)
                        {
                                append(std::make_unique<function_powell_t>(dims), dims, funcs);
                        }
                        append(std::make_unique<function_zakharov_t>(dims), dims, funcs);
                        append(std::make_unique<function_rosenbrock_t>(dims), dims, funcs);
                        append(std::make_unique<function_exponential_t>(dims), dims, funcs);
                        append(std::make_unique<function_dixon_price_t>(dims), dims, funcs);
                        append(std::make_unique<function_chung_reynolds_t>(dims), dims, funcs);
                        append(std::make_unique<function_axis_ellipsoid_t>(dims), dims, funcs);
                        append(std::make_unique<function_styblinski_tang_t>(dims), dims, funcs);
                        append(std::make_unique<function_sphere_t>(dims), dims, funcs);
                        append(std::make_unique<function_schumer_steiglitz_t>(dims), dims, funcs);
                        append(std::make_unique<function_rotated_ellipsoid_t>(dims), dims, funcs);

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
