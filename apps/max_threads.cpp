#include <thread>
#include <iostream>

int main(int, char* [])
{
        std::cout << std::thread::hardware_concurrency() << std::endl;
	return EXIT_SUCCESS;
}

