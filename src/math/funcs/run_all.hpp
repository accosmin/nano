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

namespace math
{        
        ///
        /// \brief run an operator (e.g. test, benchmark) over all test functions up to the given dimension
        ///
        template
        <
                typename tscalar,
                typename toperator
        >
        void run_all_test_functions(const typename problem_t<tscalar>::tsize max_dims, const toperator op)
        {
                op(function_beale_t<tscalar>());
                op(function_booth_t<tscalar>());
                op(function_matyas_t<tscalar>());
                op(function_colville_t<tscalar>());
                op(function_mccormick_t<tscalar>());
                op(function_himmelblau_t<tscalar>());
                op(function_rosenbrock_t<tscalar>(2));
                op(function_rosenbrock_t<tscalar>(3));
                op(function_3hump_camel_t<tscalar>());
                op(function_goldstein_price_t<tscalar>());
                op(function_bohachevsky_t<tscalar>(btype::one));
                op(function_bohachevsky_t<tscalar>(btype::two));
                op(function_bohachevsky_t<tscalar>(btype::three));
                
                for (typename math::problem_t<tscalar>::tsize dims = 1; dims <= max_dims; dims *= 2)
                {
                        op(function_trid_t<tscalar>(dims));
                        op(function_cauchy_t<tscalar>(dims));
                        op(function_sphere_t<tscalar>(dims));
                        op(function_powell_t<tscalar>(dims));
                        op(function_zakharov_t<tscalar>(dims));
                        op(function_dixon_price_t<tscalar>(dims));
                        op(function_sum_squares_t<tscalar>(dims));
                        op(function_styblinski_tang_t<tscalar>(dims));
                        op(function_rotated_ellipsoid_t<tscalar>(dims));
                }
        }
}
