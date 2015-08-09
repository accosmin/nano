#pragma once

#include "nanocv/arch.h"

namespace ncv
{        
        namespace math
        {
                ///
                /// \brief general mad-product
                ///
                template
                <
                        typename tscalar
                >
                void mad(const tscalar* NANOCV_RESTRICT idata, tscalar weight, int n, tscalar* NANOCV_RESTRICT odata)
                {
                        for (int i = 0; i < n; i ++)
                        {
                                odata[i] += idata[i] * weight;
                        }
                }

                ///
                /// \brief fixed-size mad-product
                ///
                template
                <
                        int tsize,
                        typename tscalar
                >
                void mad(const tscalar* NANOCV_RESTRICT idata, tscalar weight, tscalar* NANOCV_RESTRICT odata)
                {
                        for (int i = 0; i < tsize; i ++)
                        {
                                odata[i] += idata[i] * weight;
                        }
                }

                ///
                /// \brief unrolled mad-product
                ///
                template
                <
                        typename tscalar
                >
                void mad_unroll2(const tscalar* NANOCV_RESTRICT idata, tscalar weight, int n, tscalar* NANOCV_RESTRICT odata)
                {
                        int i = 0;
                        for ( ; i + 1 < n; i += 2)
                        {
                                odata[i + 0] += idata[i + 0] * weight;
                                odata[i + 1] += idata[i + 1] * weight;
                        }
                        switch (n % 2)
                        {
                        case 1: odata[i + 0] += idata[i + 0] * weight;
                        case 0: ;
                        }
                }

                template
                <
                        typename tscalar
                >
                void mad_unroll3(const tscalar* NANOCV_RESTRICT idata, tscalar weight, int n, tscalar* NANOCV_RESTRICT odata)
                {
                        int i = 0;
                        for ( ; i + 2 < n; i += 3)
                        {
                                odata[i + 0] += idata[i + 0] * weight;
                                odata[i + 1] += idata[i + 1] * weight;
                                odata[i + 2] += idata[i + 2] * weight;
                        }
                        switch (n % 3)
                        {
                        case 2: odata[i + 1] += idata[i + 1] * weight;
                        case 1: odata[i + 0] += idata[i + 0] * weight;
                        case 0: ;
                        }
                }

                template
                <
                        typename tscalar
                >
                void mad_unroll4(const tscalar* NANOCV_RESTRICT idata, tscalar weight, int n, tscalar* NANOCV_RESTRICT odata)
                {
                        int i = 0;
                        for ( ; i + 3 < n; i += 4)
                        {
                                odata[i + 0] += idata[i + 0] * weight;
                                odata[i + 1] += idata[i + 1] * weight;
                                odata[i + 2] += idata[i + 2] * weight;
                                odata[i + 3] += idata[i + 3] * weight;
                        }
                        switch (n % 4)
                        {
                        case 3: odata[i + 2] += idata[i + 2] * weight;
                        case 2: odata[i + 1] += idata[i + 1] * weight;
                        case 1: odata[i + 0] += idata[i + 0] * weight;
                        case 0: ;
                        }
                }

                template
                <
                        typename tscalar
                >
                void mad_unroll5(const tscalar* NANOCV_RESTRICT idata, tscalar weight, int n, tscalar* NANOCV_RESTRICT odata)
                {
                        int i = 0;
                        for ( ; i + 4 < n; i += 5)
                        {
                                odata[i + 0] += idata[i + 0] * weight;
                                odata[i + 1] += idata[i + 1] * weight;
                                odata[i + 2] += idata[i + 2] * weight;
                                odata[i + 3] += idata[i + 3] * weight;
                                odata[i + 4] += idata[i + 4] * weight;
                        }
                        switch (n % 5)
                        {
                        case 4: odata[i + 3] += idata[i + 3] * weight;
                        case 3: odata[i + 2] += idata[i + 2] * weight;
                        case 2: odata[i + 1] += idata[i + 1] * weight;
                        case 1: odata[i + 0] += idata[i + 0] * weight;
                        case 0: ;
                        }
                }

                template
                <
                        typename tscalar
                >
                void mad_unroll6(const tscalar* NANOCV_RESTRICT idata, tscalar weight, int n, tscalar* NANOCV_RESTRICT odata)
                {
                        int i = 0;
                        for ( ; i + 5 < n; i += 6)
                        {
                                odata[i + 0] += idata[i + 0] * weight;
                                odata[i + 1] += idata[i + 1] * weight;
                                odata[i + 2] += idata[i + 2] * weight;
                                odata[i + 3] += idata[i + 3] * weight;
                                odata[i + 4] += idata[i + 4] * weight;
                                odata[i + 5] += idata[i + 5] * weight;
                        }
                        switch (n % 6)
                        {
                        case 5: odata[i + 4] += idata[i + 4] * weight;
                        case 4: odata[i + 3] += idata[i + 3] * weight;
                        case 3: odata[i + 2] += idata[i + 2] * weight;
                        case 2: odata[i + 1] += idata[i + 1] * weight;
                        case 1: odata[i + 0] += idata[i + 0] * weight;
                        case 0: ;
                        }
                }

                template
                <
                        typename tscalar
                >
                void mad_unroll7(const tscalar* NANOCV_RESTRICT idata, tscalar weight, int n, tscalar* NANOCV_RESTRICT odata)
                {
                        int i = 0;
                        for ( ; i + 6 < n; i += 7)
                        {
                                odata[i + 0] += idata[i + 0] * weight;
                                odata[i + 1] += idata[i + 1] * weight;
                                odata[i + 2] += idata[i + 2] * weight;
                                odata[i + 3] += idata[i + 3] * weight;
                                odata[i + 4] += idata[i + 4] * weight;
                                odata[i + 5] += idata[i + 5] * weight;
                                odata[i + 6] += idata[i + 6] * weight;
                        }
                        switch (n % 7)
                        {
                        case 6: odata[i + 5] += idata[i + 5] * weight;
                        case 5: odata[i + 4] += idata[i + 4] * weight;
                        case 4: odata[i + 3] += idata[i + 3] * weight;
                        case 3: odata[i + 2] += idata[i + 2] * weight;
                        case 2: odata[i + 1] += idata[i + 1] * weight;
                        case 1: odata[i + 0] += idata[i + 0] * weight;
                        case 0: ;
                        }
                }

                template
                <
                        typename tscalar
                >
                void mad_unroll8(const tscalar* NANOCV_RESTRICT idata, tscalar weight, int n, tscalar* NANOCV_RESTRICT odata)
                {
                        int i = 0;
                        for ( ; i + 7 < n; i += 8)
                        {
                                odata[i + 0] += idata[i + 0] * weight;
                                odata[i + 1] += idata[i + 1] * weight;
                                odata[i + 2] += idata[i + 2] * weight;
                                odata[i + 3] += idata[i + 3] * weight;
                                odata[i + 4] += idata[i + 4] * weight;
                                odata[i + 5] += idata[i + 5] * weight;
                                odata[i + 6] += idata[i + 6] * weight;
                                odata[i + 7] += idata[i + 7] * weight;
                        }
                        switch (n % 8)
                        {
                        case 7: odata[i + 6] += idata[i + 6] * weight;
                        case 6: odata[i + 5] += idata[i + 5] * weight;
                        case 5: odata[i + 4] += idata[i + 4] * weight;
                        case 4: odata[i + 3] += idata[i + 3] * weight;
                        case 3: odata[i + 2] += idata[i + 2] * weight;
                        case 2: odata[i + 1] += idata[i + 1] * weight;
                        case 1: odata[i + 0] += idata[i + 0] * weight;
                        case 0: ;
                        }
                }
        }
}
