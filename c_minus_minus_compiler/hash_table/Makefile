TARGET = hashtable
OBJECTS = hash_table.o
CC = gcc
CFLAGS =

hashtable: hash_table.o
	$(CC) -o hashtable hash_table.o
hash_table.o: hash_table.c hash_table.h
	$(CC) $(CFLAGS) -c -g hash_table.c
clean:
	rm -f $(TARGET) $(OBJECTS)
