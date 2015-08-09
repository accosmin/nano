#pragma once

#include "nanocv/arch.h"

namespace ncv
{        
        namespace math
        {
                ///
                /// \brief general dot-product
                ///
                template
                <
                        typename tscalar
                >
                tscalar dot(const tscalar* NANOCV_RESTRICT a, const tscalar* NANOCV_RESTRICT b, int n)
                {
                        tscalar sum = 0;
                        for (int i = 0; i < n; i ++)
                        {
                                sum += a[i] * b[i];
                        }

                        return sum;
                }

                ///
                /// \brief fixed-size dot-product
                ///
                template
                <
                        int tsize,
                        typename tscalar
                >
                tscalar dot(const tscalar* NANOCV_RESTRICT a, const tscalar* NANOCV_RESTRICT b)
                {
                        tscalar sum = 0;
                        for (int i = 0; i < tsize; i ++)
                        {
                                sum += a[i] * b[i];
                        }

                        return sum;
                }

                ///
                /// \brief unrolled dot-product
                ///
                template
                <
                        typename tscalar
                >
                tscalar dot_unroll2(const tscalar* NANOCV_RESTRICT a, const tscalar* NANOCV_RESTRICT b, int n)
                {
                        int i = 0;

                        tscalar sum = 0;
                        for ( ; i + 1 < n; i += 2)
                        {
                                const tscalar sum0 = a[i + 0] * b[i + 0];
                                const tscalar sum1 = a[i + 1] * b[i + 1];
                                sum += sum0 + sum1;
                        }
                        switch (n % 2)
                        {
                        case 1: sum += a[i + 0] * b[i + 0];
                        case 0: ;
                        }

                        return sum;
                }

                template
                <
                        typename tscalar
                >
                tscalar dot_unroll3(const tscalar* NANOCV_RESTRICT a, const tscalar* NANOCV_RESTRICT b, int n)
                {
                        int i = 0;

                        tscalar sum = 0;
                        for ( ; i + 2 < n; i += 3)
                        {
                                const tscalar sum0 = a[i + 0] * b[i + 0];
                                const tscalar sum1 = a[i + 1] * b[i + 1];
                                const tscalar sum2 = a[i + 2] * b[i + 2];
                                sum += sum0 + sum1 + sum2;
                        }
                        switch (n % 3)
                        {
                        case 2: sum += a[i + 1] * b[i + 1];
                        case 1: sum += a[i + 0] * b[i + 0];
                        case 0: ;
                        }

                        return sum;
                }

                template
                <
                        typename tscalar
                >
                tscalar dot_unroll4(const tscalar* NANOCV_RESTRICT a, const tscalar* NANOCV_RESTRICT b, int n)
                {
                        int i = 0;

                        tscalar sum = 0;
                        for ( ; i + 3 < n; i += 4)
                        {
                                const tscalar sum0 = a[i + 0] * b[i + 0];
                                const tscalar sum1 = a[i + 1] * b[i + 1];
                                const tscalar sum2 = a[i + 2] * b[i + 2];
                                const tscalar sum3 = a[i + 3] * b[i + 3];
                                sum += sum0 + sum1 + sum2 + sum3;
                        }
                        switch (n % 4)
                        {
                        case 3: sum += a[i + 2] * b[i + 2];
                        case 2: sum += a[i + 1] * b[i + 1];
                        case 1: sum += a[i + 0] * b[i + 0];
                        case 0: ;
                        }

                        return sum;
                }

                template
                <
                        typename tscalar
                >
                tscalar dot_unroll5(const tscalar* NANOCV_RESTRICT a, const tscalar* NANOCV_RESTRICT b, int n)
                {
                        int i = 0;

                        tscalar sum = 0;
                        for ( ; i + 4 < n; i += 5)
                        {
                                const tscalar sum0 = a[i + 0] * b[i + 0];
                                const tscalar sum1 = a[i + 1] * b[i + 1];
                                const tscalar sum2 = a[i + 2] * b[i + 2];
                                const tscalar sum3 = a[i + 3] * b[i + 3];
                                const tscalar sum4 = a[i + 4] * b[i + 4];
                                sum += sum0 + sum1 + sum2 + sum3 + sum4;
                        }
                        switch (n % 5)
                        {
                        case 4: sum += a[i + 3] * b[i + 3];
                        case 3: sum += a[i + 2] * b[i + 2];
                        case 2: sum += a[i + 1] * b[i + 1];
                        case 1: sum += a[i + 0] * b[i + 0];
                        case 0: ;
                        }

                        return sum;
                }

                template
                <
                        typename tscalar
                >
                tscalar dot_unroll6(const tscalar* NANOCV_RESTRICT a, const tscalar* NANOCV_RESTRICT b, int n)
                {
                        int i = 0;

                        tscalar sum = 0;
                        for ( ; i + 5 < n; i += 6)
                        {
                                const tscalar sum0 = a[i + 0] * b[i + 0];
                                const tscalar sum1 = a[i + 1] * b[i + 1];
                                const tscalar sum2 = a[i + 2] * b[i + 2];
                                const tscalar sum3 = a[i + 3] * b[i + 3];
                                const tscalar sum4 = a[i + 4] * b[i + 4];
                                const tscalar sum5 = a[i + 5] * b[i + 5];
                                sum += sum0 + sum1 + sum2 + sum3 + sum4 + sum5;
                        }
                        switch (n % 6)
                        {
                        case 5: sum += a[i + 4] * b[i + 4];
                        case 4: sum += a[i + 3] * b[i + 3];
                        case 3: sum += a[i + 2] * b[i + 2];
                        case 2: sum += a[i + 1] * b[i + 1];
                        case 1: sum += a[i + 0] * b[i + 0];
                        case 0: ;
                        }

                        return sum;
                }

                template
                <
                        typename tscalar
                >
                tscalar dot_unroll7(const tscalar* NANOCV_RESTRICT a, const tscalar* NANOCV_RESTRICT b, int n)
                {
                        int i = 0;

                        tscalar sum = 0;
                        for ( ; i + 6 < n; i += 7)
                        {
                                const tscalar sum0 = a[i + 0] * b[i + 0];
                                const tscalar sum1 = a[i + 1] * b[i + 1];
                                const tscalar sum2 = a[i + 2] * b[i + 2];
                                const tscalar sum3 = a[i + 3] * b[i + 3];
                                const tscalar sum4 = a[i + 4] * b[i + 4];
                                const tscalar sum5 = a[i + 5] * b[i + 5];
                                const tscalar sum6 = a[i + 6] * b[i + 6];
                                sum += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6;
                        }
                        switch (n % 7)
                        {
                        case 6: sum += a[i + 5] * b[i + 5];
                        case 5: sum += a[i + 4] * b[i + 4];
                        case 4: sum += a[i + 3] * b[i + 3];
                        case 3: sum += a[i + 2] * b[i + 2];
                        case 2: sum += a[i + 1] * b[i + 1];
                        case 1: sum += a[i + 0] * b[i + 0];
                        case 0: ;
                        }

                        return sum;
                }

                template
                <
                        typename tscalar
                >
                tscalar dot_unroll8(const tscalar* NANOCV_RESTRICT a, const tscalar* NANOCV_RESTRICT b, int n)
                {
                        int i = 0;

                        tscalar sum = 0;
                        for ( ; i + 7 < n; i += 8)
                        {
                                const tscalar sum0 = a[i + 0] * b[i + 0];
                                const tscalar sum1 = a[i + 1] * b[i + 1];
                                const tscalar sum2 = a[i + 2] * b[i + 2];
                                const tscalar sum3 = a[i + 3] * b[i + 3];
                                const tscalar sum4 = a[i + 4] * b[i + 4];
                                const tscalar sum5 = a[i + 5] * b[i + 5];
                                const tscalar sum6 = a[i + 6] * b[i + 6];
                                const tscalar sum7 = a[i + 7] * b[i + 7];
                                sum += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
                        }
                        switch (n % 8)
                        {
                        case 7: sum += a[i + 6] * b[i + 6];
                        case 6: sum += a[i + 5] * b[i + 5];
                        case 5: sum += a[i + 4] * b[i + 4];
                        case 4: sum += a[i + 3] * b[i + 3];
                        case 3: sum += a[i + 2] * b[i + 2];
                        case 2: sum += a[i + 1] * b[i + 1];
                        case 1: sum += a[i + 0] * b[i + 0];
                        case 0: ;
                        }

                        return sum;
                }
        }
}
