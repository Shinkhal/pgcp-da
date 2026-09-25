class StateMixin:
    def ShowState(self):
        members = {mem:getattr(self,mem) for mem in vars(self)}
        print(members)
        
        
class Person:
    def __init__(self):
        self.id = 101
        self.name = "Subhani"
        self.age = 51
        
class Employee:
    def __init__(self):
        self.eid