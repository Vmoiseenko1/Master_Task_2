class MoneyBox:
    def __init__(self, capacity):
        assert ((type(capacity) is int) or (type(capacity) is float)), 'Тип числа должен быть int или float!'
        self.__volume = capacity # private characteristics
        self.__capacity = capacity # private characteristics
        
    def can_add(self, v):
        assert ((type(v) is int) or (type(v) is float)), 'Тип числа должен быть int или float!'
        if self.__capacity < v:
            return False
        else:
            return True
    
    def can_retrieve(self, v):
        assert ((type(v) is int) or (type(v) is float)), 'Тип числа должен быть int или float!'
        if v <= self.__volume - self.__capacity:
            return True
        else:
            return False 

    def add(self, v):
        if self.can_add(v) == True:
            self.__capacity -= v
    
    def retrieve(self, v):
        if self.can_retrieve(v):
            self.__capacity += v


if __name__ == "__main__":
    mb = MoneyBox(100)
    mb.add(50)
    mb.retrieve(50)