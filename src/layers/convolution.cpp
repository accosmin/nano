#include "logger.h"
#include "toeplitz.h"
#include "convolution.h"
#include "io/ibstream.h"
#include "io/obstream.h"
#include "math/random.h"
#include "math/numeric.h"
#include "tensor/numeric.h"
#include "text/to_params.h"
#include "text/from_params.h"
#include "tensor/serialize.h"

namespace nano
{
        convolution_layer_t::convolution_layer_t(const string_t& parameters) :
                layer_t(to_params(parameters, "dims", "16[1,256]", "rows", "8[1,32]", "cols", "8[1,32]",
                "conn", "1[1,16]", "drow", "1[1,8]", "dcol", "1[1,8]")),
                m_kconn(1), m_drows(1), m_dcols(1)
        {
        }

        rlayer_t convolution_layer_t::clone() const
        {
                return std::make_unique<convolution_layer_t>(*this);
        }

        tensor_size_t convolution_layer_t::resize(const tensor3d_t& tensor)
        {
                const auto idims = tensor.size<0>();
                const auto irows = tensor.size<1>();
                const auto icols = tensor.size<2>();

                const auto odims = clamp(from_params<tensor_size_t>(config(), "dims"), 1, 256);
                const auto krows = clamp(from_params<tensor_size_t>(config(), "rows"), 1, 32);
                const auto kcols = clamp(from_params<tensor_size_t>(config(), "cols"), 1, 32);
                const auto kconn = clamp(from_params<tensor_size_t>(config(), "conn"), 1, 16);
                const auto drows = clamp(from_params<tensor_size_t>(config(), "drow"), 1, 8);
                const auto dcols = clamp(from_params<tensor_size_t>(config(), "dcol"), 1, 8);

                const auto orows = (irows - krows + 1) / drows;
                const auto ocols = (icols - kcols + 1) / dcols;

                // check convolution size
                if (orows < 1 || ocols < 1)
                {
                        log_error() << "convolution layer: invalid size (" << idims << "x" << irows << "x" << icols
                                    << ") -> (" << odims << "x" << krows << "x" << kcols << ")!";
                        throw std::invalid_argument("invalid configuration for the convolution layer");
                }

                // check input connectivity factor
                if ((idims % kconn) || (odims % kconn))
                {
                        log_error() << "convolution layer: invalid input connectivity factor!";
                        throw std::invalid_argument("invalid configuration for the convolution layer");
                }

                // check stride
                if (2 * drows >= krows || 2 * dcols >= kcols)
                {
                        log_error() << "convolution layer: invalid stride - it should be less than half the kernel size!";
                        throw std::invalid_argument("invalid configuration for the convolution layer");
                }

                // resize buffers
                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);
                m_kdata.resize(odims, idims / kconn, krows, kcols);
                m_bdata.resize(odims);
                m_kconn = kconn;
                m_drows = drows;
                m_dcols = dcols;

                m_idata_toe.resize(idims, krows * kcols, orows * ocols);
                m_kdata_inv.resize(idims, odims / kconn, krows, kcols);

                m_toe_oodata.resize(odims / kconn, orows * ocols);

                m_toe_iodata.resize(krows * kcols, irows * icols);
                m_toe_iidata.resize(idims / kconn, irows * icols);

                m_toe_kodata.resize(odims / kconn, orows * ocols);
                m_toe_kkdata.resize(odims / kconn, krows * kcols);

                return psize();
        }

        void convolution_layer_t::random(scalar_t min, scalar_t max)
        {
                nano::set_random(random_t<scalar_t>(min, max), m_kdata, m_bdata);
                params_changed();
        }

        scalar_t* convolution_layer_t::save_params(scalar_t* params) const
        {
                return nano::to_array(params, m_kdata, m_bdata);
        }

        const scalar_t* convolution_layer_t::load_params(const scalar_t* params)
        {
                auto ret = nano::from_array(params, m_kdata, m_bdata);
                params_changed();
                return ret;
        }

        bool convolution_layer_t::save(obstream_t& ob) const
        {
                return  ob.write_tensor(m_kdata) &&
                        ob.write_vector(m_bdata);
        }

        bool convolution_layer_t::load(ibstream_t& ib)
        {
                return  ib.read_tensor(m_kdata) &&
                        ib.read_vector(m_bdata) &&
                        params_changed();
        }

        bool convolution_layer_t::params_changed()
        {
                for (tensor_size_t i = 0; i < imaps(); ++ i)
                {
                        for (tensor_size_t o = (i % kconn()), ok = 0; o < omaps(); ++ ok, o += kconn())
                        {
                                m_kdata_inv.vector(i, ok) = m_kdata.vector(o, i / kconn());
                        }
                }
                return true;
        }

        const tensor3d_t& convolution_layer_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.dims());

                m_idata = input;

                // bias
                for (tensor_size_t o = 0; o < omaps(); ++ o)
                {
                        m_odata.vector(o).setConstant(m_bdata(o));
                }

                // +convolution
                for (tensor_size_t i = 0; i < imaps(); ++ i)
                {
                        make_toeplitz_output(
                                m_idata.matrix(i), orows(), ocols(), krows(), kcols(), drows(), dcols(),
                                m_idata_toe.matrix(i));

                        m_toe_oodata.noalias() =
                                nano::map_matrix(m_kdata_inv.planeData(i, 0), omaps() / kconn(), krows() * kcols()) *
                                m_idata_toe.matrix(i);

                        for (tensor_size_t o = (i % kconn()), ok = 0; o < omaps(); ++ ok, o += kconn())
                        {
                                m_odata.vector(o) += m_toe_oodata.row(ok);
                        }
                }

                return m_odata;
        }

        const tensor3d_t& convolution_layer_t::ginput(const tensor3d_t& output)
        {
                assert(odims() == output.dims());

                m_odata = output;

                // correlation
                m_idata.setZero();

                for (tensor_size_t o = 0; o < omaps(); ++ o)
                {
                        make_toeplitz_ginput(
                                m_odata.matrix(o), orows(), ocols(), krows(), kcols(), drows(), dcols(), irows(), icols(),
                                m_toe_iodata);

                        m_toe_iidata.noalias() =
                                nano::map_matrix(m_kdata.planeData(o, 0), imaps() / kconn(), krows() * kcols()) *
                                m_toe_iodata;

                        for (tensor_size_t i = (o % kconn()), ik = 0; i < imaps(); ++ ik, i += kconn())
                        {
                                m_idata.vector(i) += m_toe_iidata.row(ik);
                        }
                }

                return m_idata;
        }

        void convolution_layer_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                assert(odims() == output.dims());

                m_odata = output;

                auto gkdata = nano::map_tensor(gradient, m_kdata.dims());
                auto gbdata = nano::map_vector(gradient + gkdata.size(), omaps());

                // wrt convolution
                for (tensor_size_t i = 0; i < imaps(); ++ i)
                {
                        for (tensor_size_t o = (i % kconn()), ok = 0; o < omaps(); ++ ok, o += kconn())
                        {
                                m_toe_kodata.row(ok) = m_odata.vector(o);
                        }

                        m_toe_kkdata.noalias() =
                                m_toe_kodata *
                                m_idata_toe.matrix(i).transpose();

                        for (tensor_size_t o = (i % kconn()), ok = 0; o < omaps(); ++ ok, o += kconn())
                        {
                                gkdata.vector(o, i / kconn()) = m_toe_kkdata.row(ok);
                        }
                }

                // wrt bias
                for (tensor_size_t o = 0; o < omaps(); ++ o)
                {
                        gbdata(o) = m_odata.vector(o).sum();
                }
        }
}


