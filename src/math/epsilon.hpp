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

        template <> float epsilon0<float>() { return 1e-6f; }
        template <> float epsilon1<float>() { return 1e-5f; }
        template <> float epsilon2<float>() { return 1e-4f; }
        template <> float epsilon3<float>() { return 1e-3f; }

        template <> double epsilon0<double>() { return 1e-12; }
        template <> double epsilon1<double>() { return 1e-10; }
        template <> double epsilon2<double>() { return 1e-8; }
        template <> double epsilon3<double>() { return 1e-6; }

        template <> long double epsilon0<long double>() { return 1e-14; }
        template <> long double epsilon1<long double>() { return 1e-12; }
        template <> long double epsilon2<long double>() { return 1e-10; }
        template <> long double epsilon3<long double>() { return 1e-8; }
}

