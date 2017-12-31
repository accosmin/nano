#include "test.h"

#include "trid.h"
#include "qing.h"
#include "cauchy.h"
#include "sphere.h"
#include "powell.h"
#include "sargan.h"
#include "zakharov.h"
#include "rosenbrock.h"
#include "exponential.h"
#include "dixon_price.h"
#include "axis_ellipsoid.h"
#include "chung_reynolds.h"
#include "styblinski_tang.h"
#include "rotated_ellipsoid.h"
#include "schumer_steiglitz.h"

using namespace nano;

static void append(rfunction_t&& func, const tensor_size_t dims, rfunctions_t& funcs)
{
        if (func->min_size() <= dims && dims <= func->max_size())
        {
                funcs.push_back(std::move(func));
        }
}

rfunctions_t nano::make_functions(const tensor_size_t min_size, const tensor_size_t max_size)
{
        assert(min_size >= 1);
        assert(min_size <= max_size);

        rfunctions_t funcs;
        for (tensor_size_t dims = min_size; dims <= max_size; )
        {
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

rfunctions_t nano::make_convex_functions(const tensor_size_t min_size, const tensor_size_t max_size)
{
        auto funcs = make_functions(min_size, max_size);

        funcs.erase(
                std::remove_if(funcs.begin(), funcs.end(), [] (const auto& func) { return !func->is_convex(); }),
                funcs.end());

        return funcs;
}
