#include "make_functions.h"
#include "function_trid.h"
#include "function_beale.h"
#include "function_booth.h"
#include "function_cauchy.h"
#include "function_sphere.h"
#include "function_matyas.h"
#include "function_powell.h"
#include "function_colville.h"
#include "function_zakharov.h"
#include "function_mccormick.h"
#include "function_himmelblau.h"
#include "function_rosenbrock.h"
#include "function_3hump_camel.h"
#include "function_sum_squares.h"
#include "function_dixon_price.h"
#include "function_bohachevsky.h"
#include "function_goldstein_price.h"
#include "function_styblinski_tang.h"
#include "function_rotated_ellipsoid.h"

namespace ncv
{
        namespace
        {
                void append(functions_t& dst, const functions_t& src)
                {
                        dst.insert(dst.end(), src.begin(), src.end());
                }
        }

        functions_t make_all_test_functions(const opt_size_t max_dims)
        {
                functions_t functions;

                append(functions, ncv::make_beale_funcs());
                append(functions, ncv::make_booth_funcs());
                append(functions, ncv::make_matyas_funcs());
                append(functions, ncv::make_trid_funcs(max_dims));
                append(functions, ncv::make_cauchy_funcs(max_dims));
                append(functions, ncv::make_sphere_funcs(max_dims));
                append(functions, ncv::make_powell_funcs(max_dims));
                append(functions, ncv::make_colville_funcs());
                append(functions, ncv::make_zakharov_funcs(max_dims));
                append(functions, ncv::make_mccormick_funcs());
                append(functions, ncv::make_himmelblau_funcs());
                append(functions, ncv::make_rosenbrock_funcs());
                append(functions, ncv::make_bohachevsky_funcs());
                append(functions, ncv::make_3hump_camel_funcs());
                append(functions, ncv::make_dixon_price_funcs(max_dims));
                append(functions, ncv::make_sum_squares_funcs(max_dims));
                append(functions, ncv::make_goldstein_price_funcs());
                append(functions, ncv::make_styblinski_tang_funcs(max_dims));
                append(functions, ncv::make_rotated_ellipsoid_funcs(max_dims));

                return functions;
        }
}

