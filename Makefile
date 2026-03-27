# Compiler and flags
CC = gcc
CFLAGS = -Wall -I./include

# The core library source files and their corresponding object files
SRCS = src/tensor.c src/autograd.c src/nn.c
OBJS = $(SRCS:.c=.o)

# The test executables we want to build
TEST_TENSOR = test_tensor
TEST_AUTOGRAD = test_autograd
TEST_MATMUL = test_matmul
TEST_NN = test_nn

# 'all' is the default target when you just type 'make'
all: $(TEST_TENSOR) $(TEST_AUTOGRAD) $(TEST_MATMUL) $(TEST_NN)

# Rule to compile any .c file in src/ into a .o object file
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to build the test_tensor executable
$(TEST_TENSOR): $(OBJS) tests/test_tensor.c
	$(CC) $(CFLAGS) $^ -o $@

# Rule to build the test_autograd executable
$(TEST_AUTOGRAD): $(OBJS) tests/test_autograd.c
	$(CC) $(CFLAGS) $^ -o $@

# Rule to build the test_matmul executable
$(TEST_MATMUL): $(OBJS) tests/test_matmul.c
	$(CC) $(CFLAGS) $^ -o $@

# Add this rule at the bottom for the new test:
$(TEST_NN): $(OBJS) tests/test_nn.c
	$(CC) $(CFLAGS) $^ -o $@

# Clean up build artifacts
.PHONY: clean
clean:
	rm -f src/*.o tests/*.o $(TEST_TENSOR) $(TEST_AUTOGRAD) $(TEST_MATMUL) $(TEST_NN)