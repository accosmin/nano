#include <eigen3/Eigen/Core>
#include "core/convolution.hpp"
#include <iomanip>
#include <ctime>
#include <iostream>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;
typedef std::vector<matrix_t> matrices_t;

template <typename tmatrix>
void init_matrix(int rows, int cols, tmatrix& matrix)
{
	matrix.resize(rows, cols);
	matrix.setRandom();
	matrix.array() = matrix.array().abs();
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
void test_matrices(top op, const char* name, const tmatrices& idatas, const tmatrix& kdata, tmatrix& odata)
{
        const clock_t start = clock();

	const int count = static_cast<int>(idatas.size());
	for (int i = 0; i < count; i ++)
	{
		op(idatas[i], kdata, odata);
	}

        const clock_t stop = clock();

	std::cout.precision(3);
        std::cout << name << " - " << ((stop - start + 0.0) / (CLOCKS_PER_SEC + 0.0))
                  << " (" << std::fixed << odata.sum() << ")\t";
}

void test(int isize, int ksize, int n_samples)
{
        matrices_t idatas;
        matrix_t kdata;
        matrix_t odata;

	init_matrices(isize, isize, n_samples, idatas);
	init_matrix(ksize, ksize, kdata);
	init_matrix(isize - ksize + 1, isize - ksize + 1, odata);
	kdata /= n_samples;

        test_matrices(ncv::math::conv_brut<matrix_t>,        "brt", idatas, kdata, odata);
        test_matrices(ncv::math::conv_eigen_block<matrix_t>, "eig", idatas, kdata, odata);
        test_matrices(ncv::math::conv_mod4<matrix_t>,        "md4", idatas, kdata, odata);
        test_matrices(ncv::math::conv_mod8<matrix_t>,        "md8", idatas, kdata, odata);
        test_matrices(ncv::math::conv_dynamic<matrix_t>,     "dyn", idatas, kdata, odata);
	std::cout << std::endl;
}

int main(int argc, char* argv[])
{
	static const int min_isize = 16;
	static const int max_isize = 32;
	static const int min_ksize = 8;
        static const int max_ksize = 12;
        static const int n_samples = 10000;

	for (int isize = min_isize; isize <= max_isize; isize += 2)
	{
		for (int ksize = min_ksize; ksize <= max_ksize; ksize ++)
		{
                        std::cout << "(isize = " << isize << ", ksize = " << ksize << "): \t";
                        test(isize, ksize, n_samples);
		}
                std::cout << std::endl;
	}

	return EXIT_SUCCESS;
}

