#pragma once

#include "arch.h"
#include <string>
#include <iostream>
#include "math/abs.hpp"
#include <eigen3/Eigen/Core>

std::string module_name;
std::string case_name;
std::size_t n_failures = 0;

#define ZOB_BEGIN_MODULE(name) \
int main(int, char* []) \
{ \
        module_name = #name; \
        n_failures = 0; \
        std::cout << "running test module [" << module_name << "] ..." << std::endl;

#define ZOB_CASE(name) \
        case_name = #name;

#define ZOB_END_MODULE() \
        if (n_failures > 0) \
        { \
                std::cout << "  failed with " << n_failures << " errors!" << std::endl; \
                exit(EXIT_FAILURE); \
        } \
        else \
        { \
                std::cout << "  no errors detected." << std::endl; \
                exit(EXIT_SUCCESS); \
        } \
}

#define ZOB_HANDLE_CRITICAL(critical) \
        if (critical) \
        { \
                exit(EXIT_FAILURE); \
        }
#define ZOB_HANDLE_FAILURE() \
        ++ n_failures; \
        std::cout << __FILE__ << ":" << __LINE__ << ": [" << module_name << "/" << case_name

#define ZOB_EVALUATE(check, critical) \
        if (!(check)) \
        { \
                ZOB_HANDLE_FAILURE() \
                        << "]: check {" << ZOB_STRINGIFY(check) << "} failed!" << std::endl; \
                ZOB_HANDLE_CRITICAL(critical) \
        }
#define ZOB_CHECK(check) \
        ZOB_EVALUATE(check, false)
#define ZOB_REQUIRE(check) \
        ZOB_EVALUATE(check, true)

#define ZOB_THROW(call, exception, critical) \
        try \
        { \
                call; \
                ZOB_HANDLE_FAILURE() \
                        << "]: call {" << ZOB_STRINGIFY(call) << "} does not throw!" << std::endl; \
                ZOB_HANDLE_CRITICAL(critical) \
        } \
        catch (exception&) \
        { \
        } \
        catch (...) \
        { \
                ZOB_HANDLE_FAILURE() \
                        << "]: call {" << ZOB_STRINGIFY(call) << "} does not throw {" \
                        << ZOB_STRINGIFY(exception) << "}!" << std::endl; \
                ZOB_HANDLE_CRITICAL(critical) \
        }
#define ZOB_CHECK_THROW(call, exception) \
        ZOB_THROW(call, exception, false)
#define ZOB_REQUIRE_THROW(call, exception) \
        ZOB_THROW(call, exception, true)

#define ZOB_EVALUATE_BINARY_OP(left, right, op, critical) \
        if (!((left) op (right))) \
        { \
                ZOB_HANDLE_FAILURE() \
                        << "]: check {" << ZOB_STRINGIFY(left op right) \
                        << "} failed {" << (left) << " " << ZOB_STRINGIFY(op) << " " << (right) << "}!" << std::endl; \
                ZOB_HANDLE_CRITICAL(critical) \
        }

#define ZOB_EVALUATE_EQUAL(left, right, critical) \
        ZOB_EVALUATE_BINARY_OP(left, right, ==, critical)
#define ZOB_CHECK_EQUAL(left, right) \
        ZOB_EVALUATE_EQUAL(left, right, false)
#define ZOB_REQUIRE_EQUAL(left, right) \
        ZOB_EVALUATE_EQUAL(left, right, true)

#define ZOB_EVALUATE_LESS(left, right, critical) \
        ZOB_EVALUATE_BINARY_OP(left, right, <, critical)
#define ZOB_CHECK_LESS(left, right) \
        ZOB_EVALUATE_LESS(left, right, false)
#define ZOB_REQUIRE_LESS(left, right) \
        ZOB_EVALUATE_LESS(left, right, true)

#define ZOB_EVALUATE_LESS_EQUAL(left, right, critical) \
        ZOB_EVALUATE_BINARY_OP(left, right, <=, critical)
#define ZOB_CHECK_LESS_EQUAL(left, right) \
        ZOB_EVALUATE_LESS_EQUAL(left, right, false)
#define ZOB_REQUIRE_LESS_EQUAL(left, right) \
        ZOB_EVALUATE_LESS_EQUAL(left, right, true)

#define ZOB_EVALUATE_GREATER(left, right, critical) \
        ZOB_EVALUATE_BINARY_OP(left, right, >, critical)
#define ZOB_CHECK_GREATER(left, right) \
        ZOB_EVALUATE_GREATER(left, right, false)
#define ZOB_REQUIRE_GREATER(left, right) \
        ZOB_EVALUATE_GREATER(left, right, true)

#define ZOB_EVALUATE_GREATER_EQUAL(left, right, critical) \
        ZOB_EVALUATE_BINARY_OP(left, right, >=, critical)
#define ZOB_CHECK_GREATER_EQUAL(left, right) \
        ZOB_EVALUATE_GREATER_EQUAL(left, right, false)
#define ZOB_REQUIRE_GREATER_EQUAL(left, right) \
        ZOB_EVALUATE_GREATER_EQUAL(left, right, true)

#define ZOB_EVALUATE_CLOSE(left, right, epsilon, critical) \
        ZOB_EVALUATE_LESS(zob::abs(left - right), epsilon, critical)
#define ZOB_CHECK_CLOSE(left, right, epsilon) \
        ZOB_EVALUATE_CLOSE(left, right, epsilon, false)
#define ZOB_REQUIRE_CLOSE(left, right, epsilon) \
        ZOB_EVALUATE_CLOSE(left, right, epsilon, true)

#define ZOB_EVALUATE_EIGEN_CLOSE(left, right, epsilon, critical) \
        ZOB_EVALUATE_LESS(((left - right).array().abs().maxCoeff()), epsilon, critical)
#define ZOB_CHECK_EIGEN_CLOSE(left, right, epsilon) \
        ZOB_EVALUATE_EIGEN_CLOSE(left, right, epsilon, false)
#define ZOB_REQUIRE_EIGEN_CLOSE(left, right, epsilon) \
        ZOB_EVALUATE_EIGEN_CLOSE(left, right, epsilon, true)
