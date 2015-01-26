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
                /// \brief check if two scalars are almost equal using the given precision
                ///
                template
                <
                        typename tscalar
                >
                bool close(tscalar x, tscalar y, tscalar precision)
                {
                        return x == y;
                }
                
                template <>
                inline bool close(float x, float y, float precision)
                {
                        return math::abs(x - y) <= (1 + math::abs(x) + math::abs(y)) * precision;
                }
                
                template <>
                inline bool close(double x, double y, double precision)
                {
                        return math::abs(x - y) <= (1 + math::abs(x) + math::abs(y)) * precision;
                }
                
                template <>
                inline bool close(long double x, long double y, long double precision)
                {
                        return math::abs(x - y) <= (1 + math::abs(x) + math::abs(y)) * precision;
                }

                ///
                /// \brief check if two scalars are almost equal with extremely high precision
                ///
                template
                <
                        typename tscalar
                >
                bool extremely_close(tscalar x, tscalar y)
                {
                        return close(x, y, tscalar(0));
                }

                template <>
                inline bool extremely_close(float x, float y)
                {
                        return close(x, y, std::numeric_limits<float>::epsilon());
                }

                template <>
                inline bool extremely_close(double x, double y)
                {
                        return close(x, y, std::numeric_limits<double>::epsilon());
                }

                template <>
                inline bool extremely_close(long double x, long double y)
                {
                        return close(x, y, std::numeric_limits<long double>::epsilon());
                }

                ///
                /// \brief check if two scalars are almost equal with high precision
                ///
                template
                <
                        typename tscalar
                >
                bool very_close(tscalar x, tscalar y)
                {
                        return close(x, y, tscalar(0));
                }

                template <>
                inline bool very_close(float x, float y)
                {
                        return close(x, y, std::sqrt(std::numeric_limits<float>::epsilon()));
                }

                template <>
                inline bool very_close(double x, double y)
                {
                        return close(x, y, std::sqrt(std::numeric_limits<double>::epsilon()));
                }

                template <>
                inline bool very_close(long double x, long double y)
                {
                        return close(x, y, std::sqrt(std::numeric_limits<long double>::epsilon()));
                }

                ///
                /// \brief check if two scalars are almost equal with low/relaxed precision
                ///
                template
                <
                        typename tscalar
                >
                bool quite_close(tscalar x, tscalar y)
                {
                        return close(x, y, tscalar(0));
                }

                template <>
                inline bool quite_close(float x, float y)
                {
                        return close(x, y, std::cbrt(std::numeric_limits<float>::epsilon()));
                }

                template <>
                inline bool quite_close(double x, double y)
                {
                        return close(x, y, std::cbrt(std::numeric_limits<double>::epsilon()));
                }

                template <>
                inline bool quite_close(long double x, long double y)
                {
                        return close(x, y, std::cbrt(std::numeric_limits<long double>::epsilon()));
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


                ///
                /// \brief quartic
                ///
                template
                <
                        typename tvalue
                >
                tvalue quartic(tvalue value)
                {
                        return square(square(value));
                }
        }
}

