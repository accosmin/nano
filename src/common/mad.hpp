#ifndef NANOCV_MAD_H
#define NANOCV_MAD_H

namespace ncv
{
        ///
        /// mad (multiply-add) utilities: podata += w * pidata
        ///
        namespace math
        {
                template
                <
                        typename tscalar,
                        typename tsize
                >
                void mad(const tscalar* pidata, tscalar w, tscalar* podata, tsize size)
                {
                        for (tsize k = 0; k < size; k ++)
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
                        typename tscalar,
                        typename tsize
                >
                void mad_mod4(const tscalar* pidata, tscalar w, tscalar* podata, tsize size)
                {
                        const tsize size4 = (size >> 2) << 2;

                        for (tsize k = 0; k < size4; k += 4)
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
                        typename tsize
                >
                void mad_mod4x(const tscalar* pidata, tscalar w, tscalar* podata, tsize size)
                {
                        const tsize size4 = (size >> 2) << 2;

                        for (tsize k = 0; k < size4; k += 4)
                        {
                                podata[k + 0] += w * pidata[k + 0];
                                podata[k + 1] += w * pidata[k + 1];
                                podata[k + 2] += w * pidata[k + 2];
                                podata[k + 3] += w * pidata[k + 3];
                        }

                        for (tsize k = size4; k < size; k ++)
                        {
                                podata[k] += w * pidata[k];
                        }
                }
        }
}

#endif // NANOCV_MAD_H

