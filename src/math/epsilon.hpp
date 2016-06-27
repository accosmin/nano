#pragma once

#include <cmath>
#include <limits>

namespace nano
{
        ///
        /// \brief precision level [0=very precise, 1=quite precise, 2=precise, 3=loose] for different scalars
        ///
        template
        <
                typename tscalar
        >
        tscalar epsilon0();

        template
        <
                typename tscalar
        >
        tscalar epsilon1();

        template
        <
                typename tscalar
        >
        tscalar epsilon2();

        template
        <
                typename tscalar
        >
        tscalar epsilon3();

        template <> inline float epsilon0<float>() { return 1e-6f; }
        template <> inline float epsilon1<float>() { return 1e-5f; }
        template <> inline float epsilon2<float>() { return 1e-4f; }
        template <> inline float epsilon3<float>() { return 1e-3f; }

        template <> inline double epsilon0<double>() { return 1e-12; }
        template <> inline double epsilon1<double>() { return 1e-10; }
        template <> inline double epsilon2<double>() { return 1e-8; }
        template <> inline double epsilon3<double>() { return 1e-6; }

        template <> inline long double epsilon0<long double>() { return 1e-14; }
        template <> inline long double epsilon1<long double>() { return 1e-12; }
        template <> inline long double epsilon2<long double>() { return 1e-10; }
        template <> inline long double epsilon3<long double>() { return 1e-8; }
}

