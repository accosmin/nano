#pragma once

#include "funcs/trid.hpp"
#include "funcs/beale.hpp"
#include "funcs/booth.hpp"
#include "funcs/cauchy.hpp"
#include "funcs/sphere.hpp"
#include "funcs/matyas.hpp"
#include "funcs/powell.hpp"
#include "funcs/colville.hpp"
#include "funcs/zakharov.hpp"
#include "funcs/mccormick.hpp"
#include "funcs/himmelblau.hpp"
#include "funcs/rosenbrock.hpp"
#include "funcs/3hump_camel.hpp"
#include "funcs/sum_squares.hpp"
#include "funcs/dixon_price.hpp"
#include "funcs/bohachevsky.hpp"
#include "funcs/goldstein_price.hpp"
#include "funcs/styblinski_tang.hpp"
#include "funcs/rotated_ellipsoid.hpp"

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
                                op(function_bohachevsky_t(btype::one));
                                op(function_bohachevsky_t(btype::two));
                                op(function_bohachevsky_t(btype::three));
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
