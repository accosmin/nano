#include "nanocv/tensor.h"
#include "nanocv/string.h"
#include "nanocv/tabulator.h"
#include "nanocv/measure.hpp"
#include "nanocv/math/conv2d.hpp"
#include "nanocv/math/corr2d.hpp"
#include "nanocv/math/conv3d.hpp"
#include "nanocv/math/random.hpp"
#include "nanocv/tensor/conv3d.hpp"
#include "nanocv/tensor/random.hpp"
#include <iostream>

using namespace ncv;

namespace
{
        string_t make_header(const int idims, const int isize, const int ksize, const int odims)
        {
                const int osize = isize - ksize + 1;

                return  "(" +
                        text::to_string(idims) + "x" +
                        text::to_string(isize) + "x" +
                        text::to_string(isize) + " @ " +
                        text::to_string(ksize) + "x" +
                        text::to_string(ksize) + " -> " +
                        text::to_string(odims) + "x" +
                        text::to_string(osize) + "x" +
                        text::to_string(osize) +
                        ")";
        }

        template
        <
                typename ttensor
        >
        void make_tensors(const int isize, const int idims, const int ksize, const int odims,
                ttensor& idata, ttensor& kdata, ttensor& odata)
        {
                const int osize = isize - ksize + 1;
                const int kdims = odims * idims;

                random_t<typename ttensor::Scalar> rng(-1.0 / isize, 1.0 / isize);

                idata.resize(idims, isize, isize);
                kdata.resize(kdims, ksize, ksize);
                odata.resize(odims, osize, osize);

                tensor::set_random(idata, rng);
                tensor::set_random(kdata, rng);
                tensor::set_random(odata, rng);
        }

        template
        <
                typename top,
                typename ttensori,
                typename ttensork,
                typename ttensoro
        >
        size_t measure_output(const top& op,
                const ttensori& idata, const ttensork& kdata, ttensoro&& odata, const size_t trials = 16)
        {
                return ncv::measure_robustly_usec([&] ()
                {
                        math::conv3d_output(op, idata, kdata, odata);
                }, trials);
        }

        template
        <
                typename top,
                typename ttensori,
                typename ttensork,
                typename ttensoro
        >
        size_t measure_ginput(const top& op,
                ttensori&& idata, const ttensork& kdata, const ttensoro& odata, const size_t trials = 16)
        {
                return ncv::measure_robustly_usec([&] ()
                {
                        math::conv3d_ginput(op, idata, kdata, odata);
                }, trials);
        }

        template
        <
                typename top,
                typename ttensori,
                typename ttensork,
                typename ttensoro
        >
        size_t measure_gparam(const top& op,
                const ttensori& idata, ttensork&& kdata, const ttensoro& odata, const size_t trials = 16)
        {
                return ncv::measure_robustly_usec([&] ()
                {
                        math::conv3d_gparam(op, idata, kdata, odata);
                }, trials);
        }

        void test_config_output(const int isize, const int idims, const int ksize, const int odims,
                tabulator_t::row_t& row, const size_t trials = 16)
        {
                tensor_t idata, kdata, odata;
                make_tensors(isize, idims, ksize, odims, idata, kdata, odata);

                tensor::conv3d_t<tensor_t> conv3d;
                conv3d.reset(kdata, idims, odims);

                tensor_t odata_ret = odata;

                row << measure_output(math::conv2d_eig_t(), idata, kdata, odata_ret, trials);
                row << measure_output(math::conv2d_cpp_t(), idata, kdata, odata_ret, trials);
                row << measure_output(math::conv2d_dot_t(), idata, kdata, odata_ret, trials);
                row << measure_output(math::conv2d_mad_t(), idata, kdata, odata_ret, trials);
                row << measure_output(math::conv2d_dyn_t(), idata, kdata, odata_ret, trials);
                row << ncv::measure_robustly_usec([&] () { conv3d.output(idata, odata_ret); }, trials);
        }

        void test_config_ginput(const int isize, const int idims, const int ksize, const int odims,
                tabulator_t::row_t& row, const size_t trials = 16)
        {
                tensor_t idata, kdata, odata;
                make_tensors(isize, idims, ksize, odims, idata, kdata, odata);

                tensor::conv3d_t<tensor_t> conv3d;
                conv3d.reset(kdata, idims, odims);

                tensor_t idata_ret = idata;

                row << measure_ginput(ncv::math::corr2d_egb_t(), idata_ret, kdata, odata, trials);
                row << measure_ginput(ncv::math::corr2d_egr_t(), idata_ret, kdata, odata, trials);
                row << measure_ginput(ncv::math::corr2d_cpp_t(), idata_ret, kdata, odata, trials);
                row << measure_ginput(ncv::math::corr2d_mdk_t(), idata_ret, kdata, odata, trials);
                row << measure_ginput(ncv::math::corr2d_mdo_t(), idata_ret, kdata, odata, trials);
                row << measure_ginput(ncv::math::corr2d_dyn_t(), idata_ret, kdata, odata, trials);
                row << ncv::measure_robustly_usec([&] () { conv3d.ginput(idata_ret, odata); }, trials);
        }

        void test_config_gparam(const int isize, const int idims, const int ksize, const int odims,
                tabulator_t::row_t& row, const size_t trials = 16)
        {
                tensor_t idata, kdata, odata;
                make_tensors(isize, idims, ksize, odims, idata, kdata, odata);

                tensor::conv3d_t<tensor_t> conv3d;
                conv3d.reset(kdata, idims, odims);

                tensor_t kdata_ret = kdata;

                row << measure_gparam(math::conv2d_eig_t(), idata, kdata_ret, odata, trials);
                row << measure_gparam(math::conv2d_cpp_t(), idata, kdata_ret, odata, trials);
                row << measure_gparam(math::conv2d_dot_t(), idata, kdata_ret, odata, trials);
                row << measure_gparam(math::conv2d_mad_t(), idata, kdata_ret, odata, trials);
                row << measure_gparam(math::conv2d_dyn_t(), idata, kdata_ret, odata, trials);
                row << ncv::measure_robustly_usec([&] () { conv3d.gparam(idata, kdata_ret, odata); }, trials);
        }
}

int main(int, char* [])
{
        const int min_isize = 4;
        const int max_isize = 32;

        const int min_ksize = 3;
        const int max_ksize = 9;

        const int idims = 16;
        const int odims = 32;

        const auto op_comp = [] (const string_t& value1, const string_t& value2)
        {
                return text::from_string<size_t>(value1) < text::from_string<size_t>(value2);
        };

        const auto op_marker = [=] (const strings_t& values)
        {
                return std::min_element(values.begin(), values.end(), op_comp) - values.begin();
        };

        // output
        {
                tabulator_t table("size\\output [us]");
                table.header()
                        << "2D (eig)"
                        << "2D (cpp)"
                        << "2D (dot)"
                        << "2D (mad)"
                        << "2D (dyn)"
                        << "3D (lin)";

                for (int isize = min_isize; isize <= max_isize; isize += 4)
                {
                        for (int ksize = min_ksize; ksize <= std::min(isize, max_ksize); ksize += 2)
                        {
                                const string_t header = make_header(idims, isize, ksize, odims);
                                test_config_output(isize, idims, ksize, odims, table.append(header));
                        }
                }

                table.mark(op_marker);
                table.print(std::cout);
        }

        // gradient wrt parameters
        {
                tabulator_t table("size\\gparam [us]");
                table.header()
                        << "2D (eig)"
                        << "2D (cpp)"
                        << "2D (dot)"
                        << "2D (mad)"
                        << "2D (dyn)"
                        << "3D (lin)";

                for (int isize = min_isize; isize <= max_isize; isize += 4)
                {
                        for (int ksize = min_ksize; ksize <= std::min(isize, max_ksize); ksize += 2)
                        {
                                const string_t header = make_header(idims, isize, ksize, odims);
                                test_config_gparam(isize, idims, ksize, odims, table.append(header));
                        }
                }

                table.mark(op_marker);
                table.print(std::cout);
        }

        // gradient wrt inputs
        {
                tabulator_t table("size\\ginput [us]");
                table.header()
                        << "2D (egb)"
                        << "2D (egr)"
                        << "2D (cpp)"
                        << "2D (mkd)"
                        << "2D (mko)"
                        << "2D (dyn)"
                        << "3D (lin)";

                for (int isize = min_isize; isize <= max_isize; isize += 4)
                {
                        for (int ksize = min_ksize; ksize <= std::min(isize, max_ksize); ksize += 2)
                        {
                                const string_t header = make_header(idims, isize, ksize, odims);
                                test_config_ginput(isize, idims, ksize, odims, table.append(header));
                        }
                }

                table.mark(op_marker);
                table.print(std::cout);
        }

	return EXIT_SUCCESS;
}

