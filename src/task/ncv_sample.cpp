#include "ncv_sample.h"
#include "ncv_annotation.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        isamples_t make_isamples(index_t istart, index_t icount, const irect_t& region)
        {
                isamples_t isamples(icount);
                for (index_t i = 0; i < icount; i ++)
                {
                        isamples[i].m_index = istart + i;
                        isamples[i].m_region = region;
                }

                return isamples;
        }

        //-------------------------------------------------------------------------------------------------

        void sample_t::load_gray(const annotated_image_t& aimage, const isample_t& isample)
        {
                aimage.save_gray(isample.m_region, m_data);
                load_target(aimage, isample);
        }

        //-------------------------------------------------------------------------------------------------

        void sample_t::load_rgba(const annotated_image_t& aimage, const isample_t& isample)
        {
                aimage.save_gray(isample.m_region, m_data);
                load_target(aimage, isample);
        }

        //-------------------------------------------------------------------------------------------------

        void sample_t::load_target(const annotated_image_t& aimage, const isample_t& /*isample*/)
        {
                // FIXME: assuming image classification (max one annotation for the whole image)!
                if (aimage.m_annotations.empty())
                {
                        m_target.resize(0);
                }
                else
                {
                        m_target = aimage.m_annotations[0].m_target;
                }
        }

        //-------------------------------------------------------------------------------------------------
}
