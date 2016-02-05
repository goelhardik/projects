#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include "hash_table.h"

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

/*
 * Print utility function
 */
#define FILENAME __FILE__
#define FUNCNAME __func__

void print_msg(const char *funcname, const char *msg)
{
	printf("%s in %s : %s\n", funcname, FILENAME, msg);
	return;
}

/*
 * Multiplication based hash function.
 * Hashes the key value in the given entry to an unsigned long number.
 */
unsigned long hash_table_hash(char *str)
{
	unsigned const char *key = NULL;
	unsigned long h;

	key = (unsigned const char *) str; 
	h = 0;
	while (*key != '\0') {
		h = (h * HASH_BASE + *key) % INITIAL_SIZE;
		key++;
	}
	DEBUG_PRINT(("Key = %s Hashed value = %lu\n", str, h));
	return h;
}

hash_table_t *hash_table_init(int initial_capacity, float load_factor)
{
	int i;
	hash_table_t *table = NULL;

	// allocate memory for hash table
	table = (hash_table_t *) malloc(sizeof (hash_table_t));
	if (table == NULL) {
		print_msg(FUNCNAME, "Memory allocation failed for table");
		return NULL;
	}
	memset(table, 0, sizeof (hash_table_t));
	table->capacity = initial_capacity;
	table->size = 0;
	table->load_factor = load_factor;
	DEBUG_PRINT(("Initializing hash table with capacity = %d, load_factor = %f\n", initial_capacity, load_factor));
	// allocate memory for entries in hash table
	table->entries = NULL;
	table->entries = (hash_table_entry_t **) malloc(table->capacity * sizeof (hash_table_entry_t *));
	if (table->entries == NULL) {
		print_msg(FUNCNAME, "Memory allocation failed for table entries");
		hash_table_cleanup(table);
		return NULL;
	}
	memset(table->entries, 0, sizeof (hash_table_entry_t *));
	// initialize head for all buckets 
	
	for (i = 0; i < table->capacity; i++) {
		table->entries[i] = NULL;
		table->entries[i] = (hash_table_entry_t *) malloc(sizeof (hash_table_entry_t)); 
		if (table->entries[i] == NULL) {
			print_msg(FUNCNAME, "Memory allocation failed for table entry");
			hash_table_cleanup(table);
			return NULL;
		}
		memset(table->entries[i], 0, sizeof (hash_table_entry_t));
		table->entries[i]->next == NULL;
	}

	DEBUG_PRINT(("Table initialization done.\n"));
	return table;
}

hash_table_entry_t *hash_table_create_entry(char *key)
{
	hash_table_entry_t *entry = NULL;

	entry = (hash_table_entry_t *) malloc(sizeof (hash_table_entry_t));
	if (entry == NULL) {
		print_msg(FUNCNAME, "Memory allocation failed for table entry");
		return NULL;
	}
	memset(entry, 0, sizeof (hash_table_entry_t));
	entry->key = (char *) malloc(strlen(key) + 1);
	if (entry->key == NULL) {
		print_msg(FUNCNAME, "Memory allocation failed for key");
		free(entry);
		entry = NULL;
		return NULL;
	}
	memset(entry->key, 0, sizeof (char) + 1);
	strcpy(entry->key, key);
	entry->count = 1;
	entry->next = NULL;

	return entry;
}

int hash_table_insert(hash_table_t *table, char *key)
{
	int new_size;
	unsigned long h;
	hash_table_entry_t *entry = NULL, *p = NULL;

	if (table == NULL) {
		print_msg(FUNCNAME, "NULL table passed");
		return -1;
	}
	// lookup if this key is already in the table
	p = hash_table_lookup(table, key);
	if (p) {
		DEBUG_PRINT(("Incrementing count for key = %s\n", key));
		p->count++;
		return 0;
	}

	table->size++;
	/* TODO : Implement table resizing */
	/*
	if (float(new_size / table->capacity) > table->load_factor) {
		hash_table_resize(table);
	}
	*/
	entry = hash_table_create_entry(key);
	if (entry == NULL) {
		return -1; // error message already printed while creating entry
	}
	DEBUG_PRINT(("Inserting key = %s\n", key));
	// get the bucket from hash function
	h = hash_table_hash(entry->key);
	// insert this entry at the head of the linked list
	p = table->entries[h];
	entry->next = p->next;
	p->next = entry;
	return 0;
}

int hash_table_delete(hash_table_t *table, char *key)
{
	unsigned long h;
	hash_table_entry_t *p = NULL, *pp = NULL;

	DEBUG_PRINT(("Deleting key = %s\n", key));
	// get the bucket from hash function
	h = hash_table_hash(key);
	p = table->entries[h]->next;
	pp = table->entries[h];
	while (p && (strcmp(p->key, key) != 0)) {
		pp = p;
		p = p->next;
	}
	if (p == NULL) {
		DEBUG_PRINT(("Key = %s not found\n", key));
		return -1;
	}
	pp->next = p->next;
	free(p->key);
	free(p);
	table->size--;
	return 0;
}

hash_table_entry_t *hash_table_lookup(hash_table_t *table, char *key)
{
	unsigned long h;
	hash_table_entry_t *p = NULL;

	if (table == NULL) {
		print_msg(FUNCNAME, "NULL table passed");
		return NULL;
	}
	DEBUG_PRINT(("Looking up key = %s\n", key));
	// get the bucket for this key
	h = hash_table_hash(key);
	p = table->entries[h];
	while (p->next != NULL) {
		p = p->next;
		if (strcmp(p->key, key) == 0) {
			return p;
		}
	}
	// key not found
	return NULL;
}

int free_entries(hash_table_t *table)
{
	int i;
	hash_table_entry_t *head = NULL, *p = NULL, *tmp = NULL;

	if (table->entries == NULL)
		return 0;
	/*
	 * For each bucket, iterate and free its linked list.
	 */
	for (i = 0; i < table->capacity; i++) {
		head = table->entries[i];
		p = head;
		while (p != NULL) {
			tmp = p->next;
			free(p->key);
			p->key = NULL;
			free(p);
			p = NULL;
			p = tmp;
		}
	}
	if (table->entries) {
		free(table->entries);
		table->entries = NULL;
	}

	return 0;
}

int hash_table_cleanup(hash_table_t *table)
{
	if (table == NULL) {
		return 0;
	}
	free_entries(table);
	free(table);
	table = NULL;
	return 0;
}

int comparator(const void *p, const void *q)
{
	return strcmp(((const hash_table_entry_t *)p)->key, ((const hash_table_entry_t *)q)->key);
}

void hash_table_print(hash_table_t *table)
{
	int i, c;
	hash_table_entry_t *p = NULL;
	hash_table_entry_t list[table->size];

	/*
	 * For each bucket, iterate and copy them into a list 
	 */
	c = 0;
	for (i = 0; i < table->capacity; i++) {
		p = table->entries[i]->next;
		while (p != NULL) {
			memcpy(list + c, p, sizeof (hash_table_entry_t));
			p = p->next;
			c++;
		}
	}
	// now sort the list for dictionary order
	qsort(list, table->size, sizeof (hash_table_entry_t), comparator);
	for (i = 0; i < table->size; i++) {
		printf("%s %d\n", list[i].key, list[i].count);
	}
}

/*
 * Driver code
 */
int main()
{
	hash_table_t *table = NULL;

	table = hash_table_init(INITIAL_SIZE, 1);
	hash_table_insert(table, "hardik");
	hash_table_insert(table, "goel");
	hash_table_insert(table, "goel");
	hash_table_insert(table, "goel");
	hash_table_insert(table, "koel");
	hash_table_insert(table, "loge");
	hash_table_insert(table, "goel");
	hash_table_insert(table, "hardik");
	hash_table_insert(table, "aljkjkafd");
	hash_table_insert(table, "asldfjasldkjsflasjdfashdfajkdhfkajjdsklfajslkdfjalsdkjfalkj");
	hash_table_insert(table, "asldfjasldkjsflasjdfashdfajkdhfkajjdsklfajslkdfjalsdkjfalkj");
	hash_table_insert(table, "asldfjasldkjsflasjdfashdfajkdhfkajjdsklfajslkdfjalsdkjfalkj");
	hash_table_insert(table, "asldfjasldkjsflasjdfashdfajkdhfkajjdsklfajslkdfjalsdkjfalkj");
	hash_table_insert(table, "asldfjasldkjsflasjdfashdfajkdhfkajjdsklfajslkdfjalsdkjfalkj");
	hash_table_print(table);
	hash_table_delete(table, "asldfjasldkjsflasjdfashdfajkdhfkajjdsklfajslkdfjalsdkjfalkj");
	hash_table_delete(table, "asldfjasldkjsflasjdfashdfajkdhfkajjdsklfajslkdfjalsdkjfalkj");
	hash_table_print(table);
	hash_table_insert(table, "asldfjasldkjsflasjdfashdfajkdhfkajjdsklfajslkdfjalsdkjfalkj");
	hash_table_print(table);
	hash_table_cleanup(table);
	
	return 0;
}
