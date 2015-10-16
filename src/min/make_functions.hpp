#pragma once

#include "func/function_trid.hpp"
#include "func/function_beale.hpp"
#include "func/function_booth.hpp"
#include "func/function_cauchy.hpp"
#include "func/function_sphere.hpp"
#include "func/function_matyas.hpp"
#include "func/function_powell.hpp"
#include "func/function_colville.hpp"
#include "func/function_zakharov.hpp"
#include "func/function_mccormick.hpp"
#include "func/function_himmelblau.hpp"
#include "func/function_rosenbrock.hpp"
#include "func/function_3hump_camel.hpp"
#include "func/function_sum_squares.hpp"
#include "func/function_dixon_price.hpp"
#include "func/function_bohachevsky.hpp"
#include "func/function_goldstein_price.hpp"
#include "func/function_styblinski_tang.hpp"
#include "func/function_rotated_ellipsoid.hpp"

#include <memory>

namespace func
{        
        ///
        /// \brief create all test functions up to the given dimension (if possible)
        ///
        template
        <
                typename tscalar
        >
        auto make_all_test_functions(const typename function_t<tscalar>::tsize max_dims)
        {
                typedef std::shared_ptr<function_t<tscalar>>    rfunction_t;
                typedef std::vector<rfunction_t>                functions_t;
                
                functions_t functions;

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
                
                for (typename function_t<tscalar>::tsize dims = 1; dims <= max_dims; dims *= 2)
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
