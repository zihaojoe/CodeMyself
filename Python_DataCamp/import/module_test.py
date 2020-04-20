import pandas as pd

x = {'a':[1,2]}
x = pd.DataFrame(x)
print(x)

def printHello():
	print("Hellow World")

print('End')

class Carclass(object):
    """docstring for ClassName"""
    def __init__(self, cartype, price):
        self.cartype = cartype
        self.price = price

    def printInfo(self):
        print("Type:", self.cartype)
        print("Price:", self.price)

nissan = Carclass('Nissan', 100)
nissan.printInfo()