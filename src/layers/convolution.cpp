#include "logger.h"
#include "convolution.h"
#include "math/numeric.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        convolution_layer_t::convolution_layer_t(const string_t& parameters) :
                layer_t(to_params(parameters, "dims", "16[1,256]", "rows", "8[1,32]", "cols", "8[1,32]",
                "conn", "1[1,16]", "drow", "1[1,8]", "dcol", "1[1,8]"))
        {
        }

        rlayer_t convolution_layer_t::clone() const
        {
                return std::make_unique<convolution_layer_t>(*this);
        }

        void convolution_layer_t::configure(const dim3d_t& idims)
        {
                const auto imaps = std::get<0>(idims);
                const auto irows = std::get<1>(idims);
                const auto icols = std::get<2>(idims);

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

                const auto params = conv3d_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, drows, dcols};
                m_op = conv3d_toeplitz_t{params};
        }

        void convolution_layer_t::output(tensor3d_map_t idata, tensor1d_map_t param, tensor3d_map_t odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());

                m_op.output(idata, kdata(param), bdata(param), odata);
        }

        void convolution_layer_t::ginput(tensor3d_map_t idata, tensor1d_map_t param, tensor3d_map_t odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());

                m_op.ginput(idata, kdata(param), bdata(param), odata);
        }

        void convolution_layer_t::gparam(tensor3d_map_t idata, tensor1d_map_t param, tensor3d_map_t odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());

                m_op.gparam(idata, kdata(param), bdata(param), odata);
        }
}
