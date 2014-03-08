#include "ncv.h"
#include "common/convolution.hpp"

using namespace ncv;

ncv::thread_pool_t pool;
const size_t tests = 16;

void init_matrix(int rows, int cols, matrix_t& matrix)
{
        matrix.resize(rows, cols);
        matrix.setRandom();
        matrix /= rows;
}

void init_matrices(int rows, int cols, int count, matrices_t& matrices)
{
	matrices.resize(count);
	for (int i = 0; i < count; i ++)
	{
		init_matrix(rows, cols, matrices[i]);
	}
}

void zero_matrices(matrices_t& matrices)
{
        for (size_t i = 0; i < matrices.size(); i ++)
        {
                matrices[i].setZero();
        }
}

scalar_t sum_matrices(matrices_t& matrices)
{
        scalar_t sum = 0;
        for (size_t i = 0; i < matrices.size(); i ++)
        {
                sum += matrices[i].sum();
        }
        return sum;
}

template <typename top>
scalar_t test_conv2D_1cpu(top op, const char* name, const matrices_t& idatas, const matrix_t& kdata, matrices_t& odatas)
{
        ncv::stats_t<double, size_t> proc_stats;

        // run multiple tests
        for (size_t t = 0; t < tests; t ++)
        {
                zero_matrices(odatas);

                const ncv::timer_t timer;

                for (size_t i = 0; i < idatas.size(); i ++)
                {
                        op(idatas[i], kdata, odatas[i]);
                }

                proc_stats(timer.miliseconds());
        }

        const size_t milis = static_cast<size_t>(proc_stats.avg());
        std::cout << name << "= " << text::resize(text::to_string(milis), 4, align::right) << "ms  ";

        return sum_matrices(odatas);
}

template <typename top>
scalar_t test_conv2D_xcpu(top op, const char* name, const matrices_t& idatas, const matrix_t& kdata,  matrices_t& odatas)
{
        ncv::stats_t<double, size_t> proc_stats;

        // run multiple tests
        for (size_t t = 0; t < tests; t ++)
        {
                zero_matrices(odatas);

                const ncv::timer_t timer;

                ncv::thread_loop(idatas.size(), [&] (size_t i)
                {
                        op(idatas[i], kdata, odatas[i]);
                }, pool);

                proc_stats(timer.miliseconds());
        }

        const size_t milis = static_cast<size_t>(proc_stats.avg());
        std::cout << name << "= " << text::resize(text::to_string(milis), 4, align::right) << "ms  ";

        return sum_matrices(odatas);
}

void test(int isize, int ksize, int n_samples)
{
        const int osize = isize - ksize + 1;

        matrices_t idatas, odatas;
        matrix_t kdata;

        init_matrices(isize, isize, n_samples, idatas);
        init_matrices(osize, osize, n_samples, odatas);
        init_matrix(ksize, ksize, kdata);

        std::cout << "(" << isize << "x" << isize << " @ " << ksize << "x" << ksize << "): ";
        const scalar_t sum1eib = test_conv2D_1cpu(ncv::math::conv_eib<matrix_t>, "eib(1CPU)", idatas, kdata, odatas);
        const scalar_t sumxeib = test_conv2D_xcpu(ncv::math::conv_eib<matrix_t>, "eib(xCPU)", idatas, kdata, odatas);
        const scalar_t sum1dot = test_conv2D_1cpu(ncv::math::conv_dot<matrix_t>, "dot(1CPU)", idatas, kdata, odatas);
        const scalar_t sumxdot = test_conv2D_xcpu(ncv::math::conv_dot<matrix_t>, "dot(xCPU)", idatas, kdata, odatas);
        std::cout << std::endl;

        const scalar_t eps = 1e-12;//std::numeric_limits<scalar_t>::epsilon();
        scalar_t diff = 0.0;
        if ((diff = std::fabs(sum1eib - sum1eib)) > eps) { std::cout << "eib(1CPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sumxeib - sum1eib)) > eps) { std::cout << "eib(xCPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sum1dot - sum1eib)) > eps) { std::cout << "dot(1CPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sumxdot - sum1eib)) > eps) { std::cout << "dot(xCPU) FAILED (diff = " << diff << ")!" << std::endl; }
}

int main(int argc, char* argv[])
{
        static const int min_isize = 24;
        static const int max_isize = 48;
        static const int min_ksize = 5;
        static const int max_ksize = 13;
        static const int n_samples = 4 * 1024;

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

