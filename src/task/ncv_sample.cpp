#include "ncv_sample.h"
#include "ncv_annotation.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        image_samples_t make_image_samples(index_t istart, index_t icount, index_t annotation)
        {
                image_samples_t isamples(icount);
                for (index_t i = 0; i < icount; i ++)
                {
                        isamples[i].m_image = istart + i;
                        isamples[i].m_annotation = annotation;
                }

                return isamples;
        }

        //-------------------------------------------------------------------------------------------------

        void sample_t::load_gray(const annotated_image_t& aimage, const image_sample_t& isample)
        {

        }

        //-------------------------------------------------------------------------------------------------

        void sample_t::load_rgba(const annotated_image_t& aimage, const image_sample_t& isample)
        {

        }

        //-------------------------------------------------------------------------------------------------
}
