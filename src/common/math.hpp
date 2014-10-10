#pragma once

#include <type_traits>
#include <limits>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief clamp value in the [min_value, max_value] range
                ///
                template
                <
                        typename tscalar,
                        typename tscalar_min,
                        typename tscalar_max
                >
                tscalar clamp(tscalar value, tscalar_min min_value, tscalar_max max_value)
                {
                        return  value < static_cast<tscalar>(min_value) ? static_cast<tscalar>(min_value) :
                                (value > static_cast<tscalar>(max_value) ? static_cast<tscalar>(max_value) : value);
                }

                ///
                /// \brief absolute value
                ///
                template
                <
                        typename tscalar
                >
                tscalar abs(tscalar v)
                {
                        return std::abs(v);
                }

                template <>
                inline float abs(float v)
                {
                        return std::fabs(v);
                }

                template <>
                inline double abs(double v)
                {
                        return std::fabs(v);
                }

                template <>
                inline long double abs(long double v)
                {
                        return std::fabs(v);
                }
                
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
                        return  math::abs(x - y) <=
                                (1 + math::abs(x) + math::abs(y)) * std::sqrt(std::numeric_limits<float>::epsilon());
                }
                
                template <>
                inline bool almost_equal(double x, double y)
                {
                        return  math::abs(x - y) <=
                                (1 + math::abs(x) + math::abs(y)) * std::sqrt(std::numeric_limits<double>::epsilon());
                }
                
                template <>
                inline bool almost_equal(long double x, long double y)
                {
                        return  math::abs(x - y) <=
                                (1 + math::abs(x) + math::abs(y)) * std::sqrt(std::numeric_limits<long double>::epsilon());
                }

                ///
                /// \brief square
                ///
                template
                <
                        typename tvalue
                >
                tvalue square(tvalue value)
                {
                        return value * value;
                }

                ///
                /// \brief cube
                ///
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

