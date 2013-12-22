#include <eigen3/Eigen/Core>
#include "util/convolution.hpp"
#include <iomanip>
#include <ctime>
#include <iostream>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> vector_t;
typedef std::vector<matrix_t> matrices_t;

template <typename tvector>
void init_conv1D(int size, tvector& vector)
{
        vector.resize(size);
        vector.setRandom();
}

template <typename tvector, typename tmatrix>
void init_conv2D(int rows, int cols, tvector& rvector, tvector& cvector, tmatrix& matrix)
{
        init_conv1D(rows, rvector);
        init_conv1D(cols, cvector);

        // !separable convolution!
	matrix.resize(rows, cols);
        for (int r = 0; r < rows; r ++)
        {
                for (int c = 0; c < cols; c ++)
                {
                        matrix(r, c) = rvector(r) * cvector(c);
                }
        }
}

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
void test_conv2D(top op, const char* name, const tmatrices& idatas, const tmatrix& kdata, tmatrix& odata)
{
        odata.setZero();

        const clock_t start = clock();

        const int count = static_cast<int>(idatas.size());
	for (int i = 0; i < count; i ++)
	{
                op(idatas[i], kdata, 1.0, odata);
	}

        const clock_t stop = clock();

	std::cout.precision(3);
        std::cout << name << " - " << ((stop - start + 0.0) / (CLOCKS_PER_SEC + 0.0))
                  << " (" << std::fixed << odata.sum() << ")\t";
}

template <typename tmatrices, typename tvector, typename tmatrix, typename top>
void test_sep_conv2D(top op, const char* name, const tmatrices& idatas, const tvector& krdata, const tvector& kcdata,
        tmatrix& bdata, tmatrix& odata)
{
        const clock_t start = clock();

        const int count = static_cast<int>(idatas.size());
        for (int i = 0; i < count; i ++)
        {
                op(idatas[i], krdata, kcdata, bdata, odata);
        }

        const clock_t stop = clock();

        std::cout.precision(3);
        std::cout << name << " - ";
        std::cout.precision(3);
        std::cout << ((stop - start + 0.0) / (CLOCKS_PER_SEC + 0.0));
        std::cout.precision(3);
        std::cout << " (" << std::fixed << odata.sum() << ")\t";
}

void test(int isize, int ksize, int n_samples)
{
        matrices_t idatas;
        matrix_t kdata, bdata, odata;
        vector_t krdata, kcdata;

        init_matrices(isize, isize, n_samples, idatas);
        init_matrix(isize - ksize + 1, isize - ksize + 1, odata);
        init_matrix(isize, isize, bdata);

        init_conv2D(ksize, ksize, krdata, kcdata, kdata);

        std::cout << "mix (isize = " << isize << ", ksize = " << ksize << "): \t";
        test_conv2D(ncv::math::wconv<matrix_t>,                      "org", idatas, kdata, odata);
        test_conv2D(ncv::math::wconv_mod4<matrix_t>,                 "md4", idatas, kdata, odata);
        test_conv2D(ncv::math::wconv_eigen<matrix_t>,                "eig", idatas, kdata, odata);
        if (ksize == 8)
        {
                test_conv2D(ncv::math::wconv<8, 8, matrix_t>,        "x8x", idatas, kdata, odata);
        }
        if (ksize == 16)
        {
                test_conv2D(ncv::math::wconv<16, 16, matrix_t>,      "16x", idatas, kdata, odata);
        }
        std::cout << std::endl;
}

int main(int argc, char* argv[])
{
        static const int min_isize = 24;
        static const int max_isize = 48;
	static const int min_ksize = 8;
        static const int max_ksize = 8;
        static const int n_samples = 10000;

        for (int isize = min_isize; isize <= max_isize; isize += 4)
	{
		for (int ksize = min_ksize; ksize <= max_ksize; ksize ++)
		{
                        test(isize, ksize, n_samples);
                }
	}

	return EXIT_SUCCESS;
}

