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

namespace nano
{
        enum class test_type
        {
                easy,           ///< easy test functions even for GD  (e.g. convex) - mostly useful for unit testing
                all
        };

        ///
        /// \brief run the given operator for each test function having the number of dimensions within the given range
        ///
        template
        <
                typename tscalar,
                test_type type,
                typename toperator
        >
        void foreach_test_function(
                const typename problem_t<tscalar>::tsize& min_dims,
                const typename problem_t<tscalar>::tsize& max_dims,
                const toperator& op)
        {
                if (min_dims == 1)
                {
                        switch (type)
                        {
                        case test_type::all:
                                op(function_beale_t<tscalar>());
                                op(function_booth_t<tscalar>());
                                op(function_matyas_t<tscalar>());
                                op(function_colville_t<tscalar>());
                                op(function_mccormick_t<tscalar>());
                                op(function_rosenbrock_t<tscalar>(2));
                                op(function_rosenbrock_t<tscalar>(3));
                                op(function_3hump_camel_t<tscalar>());
                                op(function_goldstein_price_t<tscalar>());
                                op(function_himmelblau_t<tscalar>());
                                op(function_bohachevsky_t<tscalar>(btype::one));
                                op(function_bohachevsky_t<tscalar>(btype::two));
                                op(function_bohachevsky_t<tscalar>(btype::three));
                                break;

                        default:
                                break;
                        }
                }

                for (typename nano::problem_t<tscalar>::tsize dims = min_dims; dims <= max_dims; dims *= 2)
                {
                        switch (type)
                        {
                        case test_type::all:
                                op(function_trid_t<tscalar>(dims));
                                op(function_powell_t<tscalar>(dims));
                                op(function_zakharov_t<tscalar>(dims));
                                op(function_dixon_price_t<tscalar>(dims));
                                op(function_styblinski_tang_t<tscalar>(dims));
                                // NB: fallthrough!

                        case test_type::easy:
                        default:

                                op(function_cauchy_t<tscalar>(dims));
                                op(function_sphere_t<tscalar>(dims));
                                op(function_sum_squares_t<tscalar>(dims));
                                op(function_rotated_ellipsoid_t<tscalar>(dims));
                                break;
                        }
                }
        }
}
