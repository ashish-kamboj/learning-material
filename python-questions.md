### Q. Which one is more memory efficient List or Tuple?
 - **Tuples do not over-allocate :** Since a tuple's size is fixed, it can be stored more compactly than lists which need to over-allocate to make append() operations efficient.
 - **Tuples refer directly to their elements :** References to objects are incorporated directly in a tuple object. In contrast, lists have an extra layer of indirection to an external array of pointers. This gives tuples a small speed advantage for indexed lookups and unpacking:

---

### Q. Undeline datastructure of set [(here)](https://stackoverflow.com/questions/3949310/how-is-set-implemented)

---

### Q. Multithreading or parallel processing in python

---

### Q. What are decorator

---

### Q. Difference between loc and iloc

---

### Q. Meaning of parameter `inplace = True` in pandas function
If it's `True` it will update the dataframe based on the function applied without assigining to object. Eg.

`df = df.sort_values('col1', inplace=False)` equivalent to `df.sort_values('col1', inplace=True)`

---

### Q. Why Generator?
Generator functions allow you to declare a function that behaves like an iterator. Any function that contains a **yield** keyword is termed a generator. Hence, yield is what makes a generator.

1. Generators allow you to create iterators in a very pythonic manner.
2. Iterators allow lazy evaluation **(saving memory space)**, only generating the next element of an iterable object when requested. This is useful for very large data sets.
3. Iterators and generators can only be iterated over once.
4. Generator Functions are better than Iterators.
5. Generator Expressions are better than Iterators (for simple cases only).

---

### Q. Difference between Array and List in Python

| List                                                      | Array
|:----------------------------------------------------------|:--------------------------------------------------------------------------------------|
| Can consist of elements belonging to different data types	| Only consists of elements belonging to the same data type
| No need to explicitly import a module for declaration	    | Need to explicitly import a module for declaration
| Cannot directly handle arithmetic operations	            | Can directly handle arithmetic operations
| Can be nested to contain different type of elements	      | Must contain either all nested elements of same size
| Preferred for shorter sequence of data items	            | Preferred for longer sequence of data items
| Greater flexibility allows easy modification (addition, deletion) of data	| Less flexibility since addition, deletion has to be done element wise
| The entire list can be printed without any explicit looping	              | A loop has to be formed to print or access the components of array
| Consume larger memory for easy addition of elements	                      | Comparatively more compact in memory size

---
