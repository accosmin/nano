#include "ncv.h"
#include "tensor/tensor_3d.hpp"
#include "tensor/tensor_4d.hpp"

int main(int argc, char* argv[])
{
        typedef ncv::tensor::tensor_3d_t<ncv::scalar_t, size_t>       tensor3d_t;
        typedef ncv::tensor::tensor_3d_t<ncv::scalar_t, size_t>       tensor4d_t;

        tensor3d_t itensor(1, 24, 24);
        tensor4d_t ktensor(8, 7, 7);
        tensor3d_t otensor(8, 18, 18);

	return EXIT_SUCCESS;
}

