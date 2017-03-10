#include "conv3d_toeplitz.h"

namespace nano
{
        conv3d_toeplitz_t::conv3d_toeplitz_t(const conv3d_params_t& params) :
                m_params(params)
        {
                const auto imaps = m_params.imaps(), irows = m_params.irows(), icols = m_params.icols();
                const auto kconn = m_params.kconn(), krows = m_params.krows(), kcols = m_params.kcols();
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols();

                // allocate buffers
                m_idata_toe.resize(imaps, krows * kcols, orows * ocols);
                m_kdata_inv.resize(imaps, omaps / kconn, krows, kcols);

                m_toe_oodata.resize(omaps / kconn, orows * ocols);

                m_toe_iodata.resize(krows * kcols, irows * icols);
                m_toe_iidata.resize(imaps / kconn, irows * icols);

                m_toe_kodata.resize(omaps / kconn, orows * ocols);
                m_toe_kkdata.resize(omaps / kconn, krows * kcols);
        }

        void conv3d_toeplitz_t::reset(const tensor4d_t& kdata) const
        {
                assert(kdata.dims() == m_params.kdims());

                const auto imaps = m_params.imaps();
                const auto kconn = m_params.kconn();
                const auto omaps = m_params.omaps();

                for (tensor_size_t i = 0; i < imaps; ++ i)
                {
                        for (tensor_size_t o = i % kconn, ok = 0; o < omaps; o += kconn, ++ ok)
                        {
                                m_kdata_inv.vector(i, ok) = kdata.vector(o, i / kconn);
                        }
                }
        }
}
