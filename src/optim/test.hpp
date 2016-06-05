#pragma once

#include "funcs/trid.h"
#include "funcs/beale.h"
#include "funcs/booth.h"
#include "funcs/cauchy.h"
#include "funcs/sphere.h"
#include "funcs/matyas.h"
#include "funcs/powell.h"
#include "funcs/colville.h"
#include "funcs/zakharov.h"
#include "funcs/mccormick.h"
#include "funcs/himmelblau.h"
#include "funcs/rosenbrock.h"
#include "funcs/3hump_camel.h"
#include "funcs/sum_squares.h"
#include "funcs/dixon_price.h"
#include "funcs/bohachevsky.h"
#include "funcs/goldstein_price.h"
#include "funcs/styblinski_tang.h"
#include "funcs/rotated_ellipsoid.h"

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
                test_type type,
                typename toperator
        >
        void foreach_test_function(
                const tensor_size_t min_dims,
                const tensor_size_t max_dims,
                const toperator& op)
        {
                if (min_dims == 1)
                {
                        switch (type)
                        {
                        case test_type::all:
                                op(function_beale_t());
                                op(function_booth_t());
                                op(function_matyas_t());
                                op(function_colville_t());
                                op(function_mccormick_t());
                                op(function_rosenbrock_t(2));
                                op(function_rosenbrock_t(3));
                                op(function_3hump_camel_t());
                                op(function_goldstein_price_t());
                                op(function_himmelblau_t());
                                op(function_bohachevsky_t(function_bohachevsky_t::btype::one));
                                op(function_bohachevsky_t(function_bohachevsky_t::btype::two));
                                op(function_bohachevsky_t(function_bohachevsky_t::btype::three));
                                break;

                        default:
                                break;
                        }
                }

                for (tensor_size_t dims = min_dims; dims <= max_dims; dims *= 2)
                {
                        switch (type)
                        {
                        case test_type::all:
                                op(function_trid_t(dims));
                                op(function_powell_t(dims));
                                op(function_zakharov_t(dims));
                                op(function_dixon_price_t(dims));
                                op(function_styblinski_tang_t(dims));
                                // NB: fallthrough!

                        case test_type::easy:
                        default:

                                op(function_cauchy_t(dims));
                                op(function_sphere_t(dims));
                                op(function_sum_squares_t(dims));
                                op(function_rotated_ellipsoid_t(dims));
                                break;
                        }
                }
        }
}
