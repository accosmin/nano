#include "common/convolution.hpp"
#include "ncv.h"

using namespace ncv;

ncv::thread_pool_t pool;

typedef double                                                                          scalar_t;
typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>        matrix_t;
typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, 1, Eigen::ColMajor>                     vector_t;
typedef std::vector<matrix_t>                                                           matrices_t;

template <typename tmatrix>
void init_matrix(int rows, int cols, tmatrix& matrix)
{
        matrix.resize(rows, cols);
        matrix.setRandom();
}

template <typename tmatrices>
void init_matrices(int rows, int cols, int count, tmatrices& matrices)
{
	matrices.resize(count);
	for (int i = 0; i < count; i ++)
	{
		init_matrix(rows, cols, matrices[i]);
	}
}

template <typename tmatrices, typename tmatrix, typename top>
void test_conv2D(top op, const char* name, const tmatrices& idatas, const tmatrix& kdata, tmatrices& odatas, bool multi)
{
        for (auto i = 0; i < idatas.size(); i ++)
        {
                odatas[i].setZero();
        }

        ncv::stats_t<double, size_t> proc_stats;

        const size_t tests = 16;
        for (size_t t = 0; t < tests; t ++)
        {
                const ncv::timer_t timer;

                if (multi)
                {
                        ncv::thread_loop(idatas.size(), [&] (size_t i)
                        {
                                op(idatas[i], kdata, odatas[i]);
                        }, pool);
                }

                else
                {
                        for (auto i = 0; i < idatas.size(); i ++)
                        {
                                op(idatas[i], kdata, odatas[i]);
                        }
                }

                proc_stats(timer.miliseconds());
        }

        typename tmatrix::Scalar sum = 0;
        for (auto i = 0; i < idatas.size(); i ++)
        {
                sum += odatas[i].sum();
        }

	using namespace ncv;
        std::cout << name
                  << "= " << text::resize(text::to_string(static_cast<size_t>(proc_stats.avg())), 6, align::right)
                  << "ms (" << text::resize(text::to_string(sum), 12, align::left) << ")\t";
}

void test(int isize, int ksize, int n_samples)
{
        const int osize = isize - ksize + 1;

        matrices_t idatas, odatas;
        matrix_t kdata;

        init_matrices(isize, isize, n_samples, idatas);
        init_matrices(osize, osize, n_samples, odatas);
        init_matrix(ksize, ksize, kdata);

        std::cout << "mix (isize = " << isize << ", ksize = " << ksize << "): \t";
        test_conv2D(ncv::math::conv_eib<matrix_t>, "eib(1CPU)", idatas, kdata, odatas, false);
        test_conv2D(ncv::math::conv_eib<matrix_t>, "eib(xCPU)", idatas, kdata, odatas, true);
        test_conv2D(ncv::math::conv_dot<matrix_t>, "dot(1CPU)", idatas, kdata, odatas, false);
        test_conv2D(ncv::math::conv_dot<matrix_t>, "dot(xCPU)", idatas, kdata, odatas, true);
        std::cout << std::endl;
}

int main(int argc, char* argv[])
{
        static const int min_isize = 24;
        static const int max_isize = 48;
        static const int min_ksize = 5;
        static const int max_ksize = 13;
        static const int n_samples = 10000;

        for (int isize = min_isize; isize <= max_isize; isize += 4)
	{
                for (int ksize = min_ksize; ksize <= max_ksize; ksize ++)
		{
                        test(isize, ksize, n_samples);
                }
                std::cout << std::endl;
	}

	return EXIT_SUCCESS;
}

