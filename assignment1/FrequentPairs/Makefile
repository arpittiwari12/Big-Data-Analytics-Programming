# DO NOT MODIFY THIS FILE
# All your code should be in `find_frequent_pairs.c`.

SHELL = /bin/sh

CC = gcc
CFLAGS = -g -Wpedantic -Wall -Wextra -std=c11 -mavx -mavx2 -O3

TARGET = target
BIN_NAME = find_frequent_pairs

$(BIN_NAME): $(TARGET)/main.o $(TARGET)/dataset.o $(TARGET)/output.o $(TARGET)/find_frequent_pairs.o
	$(CC) -o $@ $^

$(TARGET)/dataset.o: dataset.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET)/find_frequent_pairs.o: find_frequent_pairs.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET)/output.o: output.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET)/main.o: main.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET)/*.o $(BIN_NAME)

.PHONY: clean
