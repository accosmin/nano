#pragma once

#include "detail/math_impl.hpp"
#include <type_traits>
#include <limits>
#include <boost/algorithm/clamp.hpp>

namespace ncv
{
        namespace math
        {
                // forward boost functions
                using boost::algorithm::clamp;
                
                ///
                /// \brief precision comparison criteria for scalars
                ///
                template
                <
                        typename tscalar
                >
                bool almost_equal(tscalar x, tscalar y)
                {
                        return x == y;
                }
                
                template <>
                inline bool almost_equal(float x, float y)
                {
                        return std::abs(x - y) <= (1 + std::abs(x) + std::abs(y)) * std::sqrt(std::numeric_limits<float>::epsilon());
                }
                
                template <>
                inline bool almost_equal(double x, double y)
                {
                        return std::abs(x - y) <= (1 + std::abs(x) + std::abs(y)) * std::sqrt(std::numeric_limits<double>::epsilon());
                }
                
                template <>
                inline bool almost_equal(long double x, long double y)
                {
                        return std::abs(x - y) <= (1 + std::abs(x) + std::abs(y)) * std::sqrt(std::numeric_limits<long double>::epsilon());
                }

                // cast a value to another type (with rounding to the closest if necessary)
                template
                <
                        typename tround,
                        typename tvalue
                >
                tround cast(tvalue value)
                {
                        return  detail::cast<
                                tround, std::is_integral<tround>::value,
                                tvalue, std::is_integral<tvalue>::value>::dispatch(value);
                }

                // square a value
                template
                <
                        typename tvalue
                >
                tvalue square(tvalue value)
                {
                        return value * value;
                }

                template
                <
                        typename tvalue
                >
                tvalue cube(tvalue value)
                {
                        return value * square(value);
                }
        }
}

