#ifndef NANOCV_DOT_H
#define NANOCV_DOT_H

namespace ncv
{
        ///
        /// dot product utilities: sum = <pidata, pkdata>
        ///
        namespace math
        {
                template
                <
                        typename tscalar,
                        typename tsize
                >
                tscalar dot(const tscalar* pidata, const tscalar* pkdata, tsize ksize)
                {
                        tscalar sum = 0;
                        for (tsize k = 0; k < ksize; k ++)
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
                tscalar dot(const tscalar* pidata, const tscalar* pkdata, int size = 0)
                {
                        tscalar sum = 0;
                        for (int k = 0; k < tksize; k ++)
                        {
                                sum += pidata[k] * pkdata[k];
                        }

                        return sum;
                }

                template
                <
                        typename tscalar,
                        typename tsize
                >
                tscalar dot_mod4(const tscalar* pidata, const tscalar* pkdata, tsize ksize)
                {
                        const tsize ksize4 = (ksize >> 2) << 2;

                        tscalar sum = 0;
                        for (tsize k = 0; k < ksize4; k += 4)
                        {
                                sum += pidata[k + 0] * pkdata[k + 0];
                                sum += pidata[k + 1] * pkdata[k + 1];
                                sum += pidata[k + 2] * pkdata[k + 2];
                                sum += pidata[k + 3] * pkdata[k + 3];
                        }

                        return sum;
                }

                template
                <
                        typename tscalar,
                        typename tsize
                >
                tscalar dot_mod4x(const tscalar* pidata, const tscalar* pkdata, tsize ksize)
                {
                        const tsize ksize4 = (ksize >> 2) << 2;

                        tscalar sum = 0;
                        for (tsize k = 0; k < ksize4; k += 4)
                        {
                                sum += pidata[k + 0] * pkdata[k + 0];
                                sum += pidata[k + 1] * pkdata[k + 1];
                                sum += pidata[k + 2] * pkdata[k + 2];
                                sum += pidata[k + 3] * pkdata[k + 3];
                        }

                        for (tsize k = ksize4; k < ksize; k ++)
			{
                     		sum += pidata[k + 0] * pkdata[k + 0];
			}

                        return sum;
                }
        }
}

#endif // NANOCV_DOT_H

