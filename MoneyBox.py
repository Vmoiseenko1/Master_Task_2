class MoneyBox:
    def __init__(self, capacity):
        assert (type(capacity) == int() or type(capacity) == float()), 'Тип числа должен быть int или float!'
        self.__volume = capacity # private characteristics
        self.__capacity = capacity # private characteristics
        
    def can_add(self, v):
        assert (type(v) == int() or type(v) == float()), 'Тип числа должен быть int или float!'
        if self.__capacity < v:
            return False
        else:
            return True
    
    def can_retrieve(self, v):
        assert (type(v) == int() or type(v) == float()), 'Тип числа должен быть int или float!'
        if v <= self.__volume - self.__capacity:
            return True
        else:
            return False 

    def add(self, v):
        if self.__can_add(v) == True:
            self.__capacity -= v
    
    def retrieve(self, v):
        if self.__can_retrieve(v):
            self.__capacity += v
            

def main():
    mb = MoneyBox(100)
    mb.add(5)

if __name__ == "__main__":
    main()
