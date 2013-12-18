#ifndef NANOCV_DOT_H
#define NANOCV_DOT_H

#include <eigen3/Eigen/Core>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // dot product utilities:
        //      sum = <pidata, pkdata>
        /////////////////////////////////////////////////////////////////////////////////////////

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
                tscalar dot(const tscalar* pidata, const tscalar* pkdata, int)
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
                        typename tscalar
                >
                tscalar dot_mod4(const tscalar* pidata, const tscalar* pkdata, int ksize)
                {
                        const int ksize4 = (ksize >> 2) << 2;

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
                tscalar dot_eigen(const tscalar* pidata, const tscalar* pkdata, int ksize)
                {
                        typedef typename Eigen::Matrix<tscalar, Eigen::Dynamic, 1, Eigen::ColMajor> tvector;

                        const Eigen::Map<tvector> vidata(const_cast<tscalar*>(pidata), ksize);
                        const Eigen::Map<tvector> vkdata(const_cast<tscalar*>(pkdata), ksize);

                        return vidata.dot(vkdata);
                }

                template
                <
                        int tksize,
                        typename tscalar
                >
                tscalar dot_eigen(const tscalar* pidata, const tscalar* pkdata, int)
                {
                        typedef typename Eigen::Matrix<tscalar, tksize, 1, Eigen::ColMajor> tvector;

                        const Eigen::Map<tvector> vidata(const_cast<tscalar*>(pidata));
                        const Eigen::Map<tvector> vkdata(const_cast<tscalar*>(pkdata));

                        return vidata.dot(vkdata);
                }
        }
}

#endif // NANOCV_DOT_H

