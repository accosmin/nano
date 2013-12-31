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
                        typename tscalar,
			typename tindex
                >
                void mad(const tscalar* pidata, tscalar w, tscalar* podata, tindex size)
                {
                        for (tindex k = 0; k < size; k ++)
                        {
				podata[k] += w * pidata[k];
			}
                }

                template
                <
                        typename tscalar,
			typename tindex
                >
                void mad_mod4(const tscalar* pidata, tscalar w, tscalar* podata, tindex size)
                {
			const tindex size4 = (size >> 2) << 2;

                        for (tindex k = 0; k < size4; k += 4)
                        {
				podata[k + 0] += w * pidata[k + 0];
				podata[k + 1] += w * pidata[k + 1];
				podata[k + 2] += w * pidata[k + 2];
				podata[k + 3] += w * pidata[k + 3];
			}

			for (tindex k = size4; k < size; k ++)
			{				
				podata[k] += w * pidata[k];
			}
	        }

                template
                <
                        typename tscalar,
			typename tindex
                >
                void mad_mod4x(const tscalar* pidata, tscalar w, tscalar* podata, tindex size)
                {
			const tindex size4 = (size >> 2) << 2;

                        for (tindex k = 0; k < size4; k += 4)
                        {
				podata[k + 0] += w * pidata[k + 0];
				podata[k + 1] += w * pidata[k + 1];
				podata[k + 2] += w * pidata[k + 2];
				podata[k + 3] += w * pidata[k + 3];
			}
	        }

                template
                <
                        typename tscalar,
			typename tindex
                >
                void mad_eigen(const tscalar* pidata, tscalar w, tscalar* podata, tindex size)
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
                void mad_eigen(const tscalar* pidata, tscalar w, tscalar* podata)
                {
                        typedef typename Eigen::Matrix<tscalar, tsize, 1, Eigen::ColMajor> tvector;

                        const Eigen::Map<tvector> vidata(const_cast<tscalar*>(pidata));
                        const Eigen::Map<tvector> vodata(const_cast<tscalar*>(podata));

			vodata.noalias() += w * vidata;
                }
        }
}

#endif // NANOCV_MAD_H

