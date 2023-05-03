### Q. Which one is more memory efficient List or Tuple?
 - **Tuples do not over-allocate :** Since a tuple's size is fixed, it can be stored more compactly than lists which need to over-allocate to make append() operations efficient.
 - **Tuples refer directly to their elements :** References to objects are incorporated directly in a tuple object. In contrast, lists have an extra layer of indirection to an external array of pointers. This gives tuples a small speed advantage for indexed lookups and unpacking:

---

### Q. Undeline datastructure of set [(here)](https://stackoverflow.com/questions/3949310/how-is-set-implemented)

---

### Q. Multithreading or parallel processing in python

---

### Q. What are decorator
- Real world examples of decorator [(here)](https://towardsdatascience.com/decorators-in-python-advanced-8e6d3e509ffe)

---

### Q. Difference between loc and iloc

---

### Q. What *args and **kwargs actually mean [(here)](https://realpython.com/python-kwargs-and-args/)
- ***args** and ****kwargs** allow you to pass multiple arguments or keyword arguments to a function
- *args allows to pass a varying number of positional arguments
- For ***args** iterable object get using the unpacking operator * is not a list but a tuple
- ****kwargs** accepts keyword (or named) arguments instead of positional arguments
- Iterable object is standard dictionary in calse of ****kwargs**

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

### Q. Difference between Python List and Numpy Array

| List                                                                      | Array
|:--------------------------------------------------------------------------|:-----------------------------------------------------------------------|
| Can consist of elements belonging to different data types	                | Only consists of elements belonging to the same data type
| No need to explicitly import a module for declaration	                    | Need to explicitly import a module for declaration
| Cannot directly handle arithmetic operations	                             | Can directly handle arithmetic operations
| Can be nested to contain different type of elements	                      | Must contain either all nested elements of same size
| Preferred for shorter sequence of data items	                             | Preferred for longer sequence of data items
| Greater flexibility allows easy modification (addition, deletion) of data	| Less flexibility since addition, deletion has to be done element wise
| The entire list can be printed without any explicit looping	              | A loop has to be formed to print or access the components of array
| Consume larger memory for easy addition of elements	                      | Comparatively more compact in memory size

**Other Resources**
- [What are the advantages of NumPy over regular Python lists?](https://stackoverflow.com/questions/993984/what-are-the-advantages-of-numpy-over-regular-python-lists)
- [Python Lists vs. Numpy Arrays - What is the difference?](https://webcourses.ucf.edu/courses/1249560/pages/python-lists-vs-numpy-arrays-what-is-the-difference#:~:text=A%20numpy%20array%20is%20a,a%20tuple%20of%20nonnegative%20integers.&text=A%20list%20is%20the%20Python,is%20the%20real%20difference%20here.)

---

### Q. Difference between JSON and Python dictionary

|JSON                                   |Python Dictionary                                       |
|---------------------------------------|--------------------------------------------------------|
|The keys in JSON can be only strings.  |The keys in the dictionary can be any hashable object.  |
|In JSON, the keys are sequentially ordered and can be repeated.|In the dictionary, the keys cannot be repeated and must be distinct.|
|In JSON, the keys have a default value of undefined.|Dictionaries do not have any default value set.|
|IN JSON file format, the values are accessed by using the “.”(dot) or “[]” operator.|In the dictionary, the values are mostly accessed by the subscript operator. For example, if 'dict' = {'A':'123R' ,'B':'678S'} then by simply calling dict['A'] we can access values associated.|
|We are required to use the double quotation for the string object|We can use either a single or double quote for the string objects|
|The return object type in JSON is a ‘string’ object type|The return object type in a dictionary is the ‘dict’ object type|

---
