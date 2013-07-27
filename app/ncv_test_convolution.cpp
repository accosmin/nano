#include "ncv.h"
#include "core/convolution.h"
#include <iomanip>

using namespace ncv;

void init_matrix(int rows, int cols, matrix_t& matrix)
{
	matrix.resize(rows, cols);
	matrix.setRandom();
	matrix.array() = matrix.array().abs();
}

void init_matrices(int rows, int cols, int count, matrices_t& matrices)
{
	matrices.resize(count);
	for (int i = 0; i < count; i ++)
	{
		init_matrix(rows, cols, matrices[i]);
	}
}

template <typename top>
void test_matrices(top op, const char* name, const matrices_t& idatas, const matrix_t& kdata, matrix_t& odata)
{
        const clock_t start = clock();

	odata.setZero();

	const int count = static_cast<int>(idatas.size());
	for (int i = 0; i < count; i ++)
	{
		op(idatas[i], kdata, odata);
	}

        const clock_t stop = clock();

        std::cout.precision(3);
	std::cout << std::fixed << odata.sum();
	std::cout.precision(3);
        std::cout << " (" << name << " - " << ((stop - start + 0.0) / (CLOCKS_PER_SEC + 0.0)) << ")\t";
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

        test_matrices(math::conv_add_naive<matrix_t>,       "naive  ", idatas, kdata, odata);
        test_matrices(math::conv_add_eigen_block<matrix_t>, "eigen  ", idatas, kdata, odata);
        test_matrices(math::conv_add_mod4<matrix_t>,        "mod4   ", idatas, kdata, odata);
        test_matrices(math::conv_add_mod8<matrix_t>,        "mod8   ", idatas, kdata, odata);
        test_matrices(math::conv_add_dynamic<matrix_t>,     "dynamic", idatas, kdata, odata);
	std::cout << std::endl;
}

int main(int argc, char* argv[])
{
	static const int min_isize = 16;
	static const int max_isize = 32;
	static const int min_ksize = 8;
	static const int max_ksize = 12;
        static const int n_samples = 100000;

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

