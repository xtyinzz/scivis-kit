CC = g++
CFLAGS=-I. -MD -MP
SRC_DIR = src
DEPS = $(wildcard $(SRC_DIR)/%.h)
SRC = $(wildcard $(SRC_DIR)/%.cpp$)


# %: %.o
# 	g++ $< -o $@

all: task1a task1b task1c task2a task2b task2c

$(SRC_DIR)%.o: $(SRC_DIR)%.cpp
	g++ -c $< -o $@


task1a: $(SRC_DIR)/task1a.o
	g++ $< -o $@

task1b: $(SRC_DIR)/task1b.o
	g++ $< -o $@

task1c: $(SRC_DIR)/task1c.o
	g++ $< -o $@

task2a: $(SRC_DIR)/task2a.o
	g++ $< -o $@

task2b: $(SRC_DIR)/task2b.o
	g++ $< -o $@

task2c: $(SRC_DIR)/task2c.o
	g++ $< -o $@


clean:
	-del $(SRC_DIR)\*.o *.exe

# task1b: $(SRC_DIR)/task1b.cpp
# 	g++ $(SRC_DIR)/task1a.cpp -o task1a

# task1c: $(SRC_DIR)/task1c.cpp
# 	g++ $(SRC_DIR)/task1a.cpp -o task1a



# task1b: task1b.o
# task1c: task1c.o

# task1a.o: task1a.cpp
# task1b.o: task1c.cpp
# task1c.o: task1c.cpp