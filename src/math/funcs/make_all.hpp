#pragma once

#include "function_trid.hpp"
#include "function_beale.hpp"
#include "function_booth.hpp"
#include "function_cauchy.hpp"
#include "function_sphere.hpp"
#include "function_matyas.hpp"
#include "function_powell.hpp"
#include "function_colville.hpp"
#include "function_zakharov.hpp"
#include "function_mccormick.hpp"
#include "function_himmelblau.hpp"
#include "function_rosenbrock.hpp"
#include "function_3hump_camel.hpp"
#include "function_sum_squares.hpp"
#include "function_dixon_price.hpp"
#include "function_bohachevsky.hpp"
#include "function_goldstein_price.hpp"
#include "function_styblinski_tang.hpp"
#include "function_rotated_ellipsoid.hpp"
#include <vector>
#include <memory>

namespace math
{        
        ///
        /// \brief collects all test functions up to the given dimension
        ///
        template
        <
                typename tscalar
        >
        auto make_all_test_functions(
                const typename problem_t<tscalar>::tsize min_dims,
                const typename problem_t<tscalar>::tsize max_dims)
        {
                std::vector<std::shared_ptr<function_t<tscalar>>> functions;

                if (min_dims <= 1)
                {
                        functions.push_back(std::make_shared<function_beale_t<tscalar>>());
                        functions.push_back(std::make_shared<function_booth_t<tscalar>>());
                        functions.push_back(std::make_shared<function_matyas_t<tscalar>>());
                        functions.push_back(std::make_shared<function_colville_t<tscalar>>());
                        functions.push_back(std::make_shared<function_mccormick_t<tscalar>>());
                        functions.push_back(std::make_shared<function_himmelblau_t<tscalar>>());
                        functions.push_back(std::make_shared<function_rosenbrock_t<tscalar>>(2));
                        functions.push_back(std::make_shared<function_rosenbrock_t<tscalar>>(3));
                        functions.push_back(std::make_shared<function_3hump_camel_t<tscalar>>());
                        functions.push_back(std::make_shared<function_goldstein_price_t<tscalar>>());
                        functions.push_back(std::make_shared<function_bohachevsky_t<tscalar>>(btype::one));
                        functions.push_back(std::make_shared<function_bohachevsky_t<tscalar>>(btype::two));
                        functions.push_back(std::make_shared<function_bohachevsky_t<tscalar>>(btype::three));
                }
                
                for (typename math::problem_t<tscalar>::tsize dims = min_dims; dims <= max_dims; dims *= 2)
                {
                        functions.push_back(std::make_shared<function_trid_t<tscalar>>(dims));
                        functions.push_back(std::make_shared<function_cauchy_t<tscalar>>(dims));
                        functions.push_back(std::make_shared<function_sphere_t<tscalar>>(dims));
                        functions.push_back(std::make_shared<function_powell_t<tscalar>>(dims));
                        functions.push_back(std::make_shared<function_zakharov_t<tscalar>>(dims));
                        functions.push_back(std::make_shared<function_dixon_price_t<tscalar>>(dims));
                        functions.push_back(std::make_shared<function_sum_squares_t<tscalar>>(dims));
                        functions.push_back(std::make_shared<function_styblinski_tang_t<tscalar>>(dims));
                        functions.push_back(std::make_shared<function_rotated_ellipsoid_t<tscalar>>(dims));
                }

                return functions;
        }
}
