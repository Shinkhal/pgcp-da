#high order functions

import time 

# def introspect_time(fnc):
#     def inner_fun(*arg, **kwargs):
#         start = time.time()
#         result = fnc(*arg,**kwargs)#callback
#         end = time.time()
#         print(f"Time Taken is {end-start}")
#         return result
#     return inner_fun

class introspect_time:
    def __init__(self,fnc):
        self.fnc = fnc
        
    def __call__(self, *args, **kwds):
        start = time.time()
        result = self.fnc(*args,**kwds)
        end = time.time()
        print(f"Time taken in Class : [ {end - start}]")
        return result

print(time.time())


@introspect_time
def add_fun(x,y):
    time.sleep(1)
    return x + y



@introspect_time
def print_list(lst):
    for item in lst:
        time.sleep(1)
        print(f"{item}",end="\t-\t")
        
    print()
    
res = add_fun(20,10)
print("Result add ",res , sep="\t:\t")
print("=" * 60)

lst = [x**2 for x in range(1,6)]
print_list(lst)