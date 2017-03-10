#include "logger.h"
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
                const auto imaps = tensor.size<0>();
                const auto irows = tensor.size<1>();
                const auto icols = tensor.size<2>();

                const auto omaps = clamp(from_params<tensor_size_t>(config(), "dims"), 1, 256);
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
                        log_error() << "convolution layer: invalid size (" << imaps << "x" << irows << "x" << icols
                                    << ") -> (" << omaps << "x" << krows << "x" << kcols << ")!";
                        throw std::invalid_argument("invalid configuration for the convolution layer");
                }

                // check input connectivity factor
                if ((imaps % kconn) || (omaps % kconn))
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
                m_idata.resize(imaps, irows, icols);
                m_odata.resize(omaps, orows, ocols);
                m_kdata.resize(omaps, imaps / kconn, krows, kcols);
                m_bdata.resize(omaps);

                m_kconn = kconn;
                m_drows = drows;
                m_dcols = dcols;

                const auto params = conv3d_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, drows, dcols};
                m_op = conv3d_toeplitz_t{params};

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
                m_op.reset(m_kdata);
                return true;
        }

        const tensor3d_t& convolution_layer_t::output(const tensor3d_t& input)
        {
                m_idata = input;
                m_op.output(m_idata, m_kdata, m_bdata, m_odata);
                return m_odata;
        }

        const tensor3d_t& convolution_layer_t::ginput(const tensor3d_t& output)
        {
                m_odata = output;
                m_op.ginput(m_idata, m_kdata, m_bdata, m_odata);
                return m_idata;
        }

        void convolution_layer_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                m_odata = output;
                auto gkdata = map_tensor(gradient, m_kdata.dims());
                auto gbdata = map_vector(gradient + gkdata.size(), omaps());
                m_op.gparam(m_idata, gkdata, gbdata, m_odata);
        }
}
