#ifndef NANOCV_MAD_H
#define NANOCV_MAD_H

#include <eigen3/Eigen/Core>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // mad utilities:
        //      podata += w * pidata
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                template
                <
                        typename tscalar
                >
                void mad(const tscalar* pidata, tscalar w, tscalar* podata, int size)
                {
                        for (int k = 0; k < size; k ++)
                        {
				podata[k] += w * pidata[k];
			}
                }

                template
                <
                        int tsize,
                        typename tscalar
                >
                void mad(const tscalar* pidata, tscalar w, tscalar* podata, int size = 0)
                {
                        for (int k = 0; k < tsize; k ++)
                        {
                                podata[k] += w * pidata[k];
                        }
                }

                template
                <
                        typename tscalar
                >
                void mad_mod4(const tscalar* pidata, tscalar w, tscalar* podata, int size)
                {
                        const int size4 = (size >> 2) << 2;

                        for (int k = 0; k < size4; k += 4)
                        {
				podata[k + 0] += w * pidata[k + 0];
				podata[k + 1] += w * pidata[k + 1];
				podata[k + 2] += w * pidata[k + 2];
				podata[k + 3] += w * pidata[k + 3];
                        }
	        }

                template
                <
                        typename tscalar
                >
                void mad_mod4x(const tscalar* pidata, tscalar w, tscalar* podata, int size)
                {
                        const int size4 = (size >> 2) << 2;

                        for (int k = 0; k < size4; k += 4)
                        {
                                podata[k + 0] += w * pidata[k + 0];
                                podata[k + 1] += w * pidata[k + 1];
                                podata[k + 2] += w * pidata[k + 2];
                                podata[k + 3] += w * pidata[k + 3];
                        }

                        for (int k = size4; k < size; k ++)
                        {
                                podata[k] += w * pidata[k];
                        }
                }

                template
                <
                        typename tscalar
                >
                void mad_eig(const tscalar* pidata, tscalar w, tscalar* podata, int size)
                {
                        typedef typename Eigen::Matrix<tscalar, Eigen::Dynamic, 1, Eigen::ColMajor> tvector;

                        const Eigen::Map<tvector> vidata(const_cast<tscalar*>(pidata), size);
                        const Eigen::Map<tvector> vodata(const_cast<tscalar*>(podata), size);

			vodata.noalias() += w * vidata;
                }

                template
                <
			int tsize,
                        typename tscalar
                >
                void mad_eig(const tscalar* pidata, tscalar w, tscalar* podata)
                {
                        typedef typename Eigen::Matrix<tscalar, tsize, 1, Eigen::ColMajor> tvector;

                        const Eigen::Map<tvector> vidata(const_cast<tscalar*>(pidata));
                        const Eigen::Map<tvector> vodata(const_cast<tscalar*>(podata));

			vodata.noalias() += w * vidata;
                }
        }
}

#endif // NANOCV_MAD_H

