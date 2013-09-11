#ifndef NANOCV_MADD_H
#define NANOCV_MADD_H

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // multiply-add vectorial utilities:
        //      pidata = coeff * pkdata
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                template
                <
                        typename tscalar
                >
                void mad(tscalar* pidata, tscalar coeff, const tscalar* pkdata, int size)
                {
                        for (int i = 0; i < size; i ++)
                        {
                                pidata[i] += coeff * pkdata[i];
                        }
                }

                template
                <
                        int tsize,
                        typename tscalar
                >
                void mad(tscalar* pidata, tscalar coeff, const tscalar* pkdata)
                {
                        mad(pidata, coeff, pkdata, tsize);
                }

                template
                <
                        typename tscalar
                >
                void mad_mod4(tscalar* pidata, tscalar coeff, const tscalar* pkdata, int size4, int size)
                {
                        for (int i = 0; i < size4; i += 4)
                        {
                                pidata[i + 0] += coeff * pkdata[i + 0];
                                pidata[i + 1] += coeff * pkdata[i + 1];
                                pidata[i + 2] += coeff * pkdata[i + 2];
                                pidata[i + 3] += coeff * pkdata[i + 3];
                        }
                        for (int i = size4; i < size; i ++)
                        {
                                pidata[i + 0] += coeff * pkdata[i + 0];
                        }
                }

                template
                <
                        typename tscalar
                >
                void mad_mod8(tscalar* pidata, tscalar coeff, const tscalar* pkdata, int size8, int size)
                {
                        for (int i = 0; i < size8; i += 8)
                        {
                                pidata[i + 0] += coeff * pkdata[i + 0];
                                pidata[i + 1] += coeff * pkdata[i + 1];
                                pidata[i + 2] += coeff * pkdata[i + 2];
                                pidata[i + 3] += coeff * pkdata[i + 3];
                                pidata[i + 4] += coeff * pkdata[i + 4];
                                pidata[i + 5] += coeff * pkdata[i + 5];
                                pidata[i + 6] += coeff * pkdata[i + 6];
                                pidata[i + 7] += coeff * pkdata[i + 7];
                        }
                        for (int i = size8; i < size; i ++)
                        {
                                pidata[i + 0] += coeff * pkdata[i + 0];
                        }
                }
        }
}

#endif // NANOCV_MADD_H

