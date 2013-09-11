#ifndef NANOCV_DOT_H
#define NANOCV_DOT_H

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // dot product utilities:
        //      sum = <pidata, pkdata>
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                template
                <
                        typename tscalar
                >
                tscalar dot(const tscalar* pidata, const tscalar* pkdata, int ksize)
                {
                        tscalar sum = 0;
                        for (int k = 0; k < ksize; k ++)
                        {
                                sum += pidata[k] * pkdata[k];
                        }

                        return sum;
                }

                template
                <
                        int tksize,
                        typename tscalar
                >
                tscalar dot(const tscalar* pidata, const tscalar* pkdata)
                {
                        return dot(pidata, pkdata, tksize);
                }

                template
                <
                        typename tscalar
                >
                tscalar dot_mod4(const tscalar* pidata, const tscalar* pkdata, int ksize4, int ksize)
                {
                        tscalar sum = 0;
                        for (int k = 0; k < ksize4; k += 4)
                        {
                                sum += pidata[k + 0] * pkdata[k + 0];
                                sum += pidata[k + 1] * pkdata[k + 1];
                                sum += pidata[k + 2] * pkdata[k + 2];
                                sum += pidata[k + 3] * pkdata[k + 3];
                        }
                        for (int k = ksize4; k < ksize; k ++)
                        {
                                sum += pidata[k + 0] * pkdata[k + 0];
                        }

                        return sum;
                }

                template
                <
                        typename tscalar
                >
                tscalar dot_mod8(const tscalar* pidata, const tscalar* pkdata, int ksize8, int ksize)
                {
                        tscalar sum = 0;
                        for (int k = 0; k < ksize8; k += 8)
                        {
                                sum += pidata[k + 0] * pkdata[k + 0];
                                sum += pidata[k + 1] * pkdata[k + 1];
                                sum += pidata[k + 2] * pkdata[k + 2];
                                sum += pidata[k + 3] * pkdata[k + 3];
                                sum += pidata[k + 4] * pkdata[k + 4];
                                sum += pidata[k + 5] * pkdata[k + 5];
                                sum += pidata[k + 6] * pkdata[k + 6];
                                sum += pidata[k + 7] * pkdata[k + 7];
                        }
                        for (int k = ksize8; k < ksize; k ++)
                        {
                                sum += pidata[k + 0] * pkdata[k + 0];
                        }

                        return sum;
                }
        }
}

#endif // NANOCV_DOT_H

