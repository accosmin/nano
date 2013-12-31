#include "util/dot.hpp"
#include "util/convolution.hpp"
#include <iomanip>
#include <ctime>
#include <iostream>
#include <functional>

typedef double                                                                          scalar_t;
typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>        matrix_t;
typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, 1, Eigen::ColMajor>                     vector_t;
typedef std::vector<matrix_t>                                                           matrices_t;

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

template <typename tmatrices, typename tmatrix, typename top>
void test_conv2D_dot(top op, const char* name, const tmatrices& idatas, const tmatrix& kdata, tmatrix& odata)
{
        odata.setZero();

        const clock_t start = clock();

        const int count = static_cast<int>(idatas.size());
        for (int i = 0; i < count; i ++)
        {
                ncv::math::wconv_dot<true>(idatas[i], kdata, 1.0, odata, op);
        }

        const clock_t stop = clock();

        std::cout.precision(3);
        std::cout << name << " - " << ((stop - start + 0.0) / (CLOCKS_PER_SEC + 0.0))
                  << " (" << std::fixed << odata.sum() << ")\t";
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
        test_conv2D_dot(ncv::math::dot<scalar_t, matrix_t::Index>,              "org", idatas, kdata, odata);
        test_conv2D_dot(ncv::math::dot_mod4<scalar_t, matrix_t::Index>,         "m4-", idatas, kdata, odata);
	if (kdata.cols() % 4 == 0)
	{
                test_conv2D_dot(ncv::math::dot_mod4x<scalar_t, matrix_t::Index>,"m4x", idatas, kdata, odata);
        }
        test_conv2D_dot(ncv::math::dot_eig<scalar_t, matrix_t::Index>,          "eig", idatas, kdata, odata);
        test_conv2D(ncv::math::wconv_eib<true, matrix_t>,                       "eib", idatas, kdata, odata);
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

