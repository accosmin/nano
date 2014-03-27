#ifndef NANOCV_SUM_H
#define NANOCV_SUM_H

namespace ncv
{
        ///
        /// sum utilities: sum(pdata, i=1,size)
        ///
        namespace math
        {
                template
                <
                        typename tscalar,
                        typename tsize
                >
                tscalar sum(const tscalar* pdata, tsize ksize)
                {
                        tscalar ret = 0;
                        for (tsize k = 0; k < ksize; k ++)
                        {
                                ret += pdata[k];
                        }

                        return ret;
                }

                template
                <
                        int tksize,
                        typename tscalar
                >
                tscalar sum(const tscalar* pdata, int size = 0)
                {
                        tscalar ret = 0;
                        for (int k = 0; k < tksize; k ++)
                        {
                                ret += pdata[k];
                        }

                        return ret;
                }

                template
                <
                        typename tscalar,
                        typename tsize
                >
                tscalar sum_mod4(const tscalar* pdata, tsize ksize)
                {
                        const tsize ksize4 = (ksize >> 2) << 2;

                        tscalar ret = 0;
                        for (tsize k = 0; k < ksize4; k += 4)
                        {
                                ret += pdata[k + 0];
                                ret += pdata[k + 1];
                                ret += pdata[k + 2];
                                ret += pdata[k + 3];
                        }

                        for (tsize k = ksize4; k < ksize; k ++)
			{
                                ret += pdata[k + 0];
			}

                        return ret;
                }
        }
}

#endif // NANOCV_SUM_H

