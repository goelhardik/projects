#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#define HASH_BASE 256 
#define INITIAL_SIZE 97

/* Hash table structures */
typedef struct hash_table_entry_s {
	char *key;
	int count;
	struct hash_table_entry_s *next;
} hash_table_entry_t;

typedef struct hash_table_s {
	int capacity;
	int size;
	float load_factor;
	hash_table_entry_t **entries;
} hash_table_t;

/* Hash table management routines */
unsigned long hash_table_hash(char *str);
hash_table_t *hash_table_init(int initial_capacity, float load_factor);
hash_table_entry_t *hash_table_create_entry(char *key);
int hash_table_insert(hash_table_t *table, char *key);
int hash_table_delete(hash_table_t *table, char *key);
hash_table_entry_t *hash_table_lookup(hash_table_t *table, char *key);
int hash_table_cleanup(hash_table_t *table);
void hash_table_print(hash_table_t *table);
#endif
