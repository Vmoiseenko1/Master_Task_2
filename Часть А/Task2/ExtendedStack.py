class ExtendedStack(list): # Наследование от list
    def sum(self):
        if len(self) >= 2:
            self.append(self.pop() + self.pop())
    def sub(self):
        if len(self) >= 2:
            self.append(self.pop() - self.pop())
    def mul(self):
        if len(self) >= 2:
            self.append(self.pop() * self.pop())
    def div(self):
        if len(self) >= 2:
            pop1 = self.pop()
            pop2 = self.pop()
            if pop2 != 0:
                div = pop1 // pop2
                self.append(div)
