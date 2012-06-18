class A:
    def __init__(self):
        self.val = "class A"
        
class C:
    def __init__(self):
        self.val = "class C"
        self.b = B()
        
class B:
    def __init__(self):
        self.val = "class B"
        self.a = A()
        
c=C()
print c.val
print c.b
