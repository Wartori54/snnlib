# makefile

.PHONY: clean test all scratches

CC = g++
CFLAGS = -Wall -Wextra -Wpedantic -fPIC -g -O2

SRC_DIR=./src
INC_DIR=./include
OBJ_DIR=./obj
OUT_DIR=./lib
TST_DIR=./test

OUT_FILE_NAME=libnn.a
TEST_FILE_NAME=main

SRCS=$(wildcard $(SRC_DIR)/*.cpp)
OBJS=$(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

all: $(OUT_DIR)/$(OUT_FILE_NAME)
	@echo "Done!"

$(OUT_DIR)/$(OUT_FILE_NAME): $(OBJS)
	mkdir -p $(OBJ_DIR)
	ar rs $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(INC_DIR)/%.h
	mkdir -p $(OBJ_DIR)
	$(CC) -c -I$(INC_DIR) $(CFLAGS) -o $@ $<

#$(SRC_DIR)/%.cpp: $(INC_DIR)/%.h

dirmake:
	mkdir -p $(OUT_DIR)
	mkdir -p $(OBJ_DIR)

test: $(TST_DIR)/$(TEST_FILE_NAME).cpp all
	$(CC) -I$(INC_DIR) $(CFLAGS) -o $(TST_DIR)/$(TEST_FILE_NAME) $< $(OUT_DIR)/$(OUT_FILE_NAME)

scratches: $(TST_DIR)/scratches.cpp
	$(CC) $(CFLAGS) -o $(TST_DIR)/scratches $<